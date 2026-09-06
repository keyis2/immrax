from functools import wraps
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
from jax._src.debugging import debug_callback_p
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax._src import ad_util, config, source_info_util
from jax._src.core import (
    Atom,
    Jaxpr,
    Literal,
    Var,
    clean_up_dead_vars,
    last_used,
    typecheck,
)
from jax._src.util import safe_map
from jax.extend.core import Primitive
from jax._src.lax import linalg as LA

# TODO: import only necessary things
from immrax.inclusion.interval import Interval, interval
from functools import partial

"""
This file implements the Natural Inclusion Function as an interpreter of Jaxprs.
"""

inclusion_registry = {}


def natif(
    f: Callable[..., jax.Array], *, fixed_argnums: int | Sequence[int] = None
) -> Callable[..., Interval]:
    """Creates a Natural Inclusion Function of f.

    All (non-fixed) positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Natural Inclusion Function from
    fixed_argnums : int|Sequence[int]
        Positional arguments to be treated as jax.Array instead of Interval

    Returns
    -------
    Callable[..., Interval]
        Natural Inclusion Function of f

    """

    @jit
    @wraps(f)
    def wrapped(*args, **kwargs):
        """Natural inclusion function.
        """
        # Traverse the args and kwargs, replacing intervals with lower bounds.
        # Convert args to at least jax.Array when they are not interval
        getlower = lambda x: x.lower if isinstance(x, Interval) else jnp.asarray(x)
        isinterval = lambda x: isinstance(x, Interval)
        buildargs = jax.tree_util.tree_map(getlower, args, is_leaf=isinterval)
        # kwargs stay not jax.Array
        getlower = lambda x: x.lower if isinstance(x, Interval) else x
        buildkwargs = jax.tree_util.tree_map(getlower, kwargs, is_leaf=isinterval)
        # Build a jaxpr via evaluation on the lower bounds only. TODO: Do we need eqx.filter_make_jaxpr?
        # closed_jaxpr = jax.make_jaxpr(f)(*buildargs, **buildkwargs)
        closed_jaxpr = eqx.filter_make_jaxpr(f)(*buildargs, **buildkwargs)[0]
        # Evaluate the jaxpr on the interval arguments using natif_jaxpr.
        out = natif_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        if len(out) == 1:
            return out[0]
        return out

    return wrapped

def _has_nan(x):
    if isinstance(x, Interval):
        return jnp.any(jnp.isnan(x.lower)) | jnp.any(jnp.isnan(x.upper))
    elif isinstance(x, (jax.Array, jnp.ndarray)):
        return jnp.any(jnp.isnan(x))
    else:
        return False

def _has_inf(x):
    if isinstance(x, Interval):
        return jnp.any(jnp.isinf(x.lower)) | jnp.any(jnp.isinf(x.upper))
    elif isinstance(x, (jax.Array, jnp.ndarray)):
        return jnp.any(jnp.isinf(x))
    else:
        return False

def natif_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)
    for i, eqn in enumerate(jaxpr.eqns):
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any([isinstance(read(iv), Interval) for iv in eqn.invars]):
                try:
                    ans = inclusion_registry[eqn.primitive](
                        *subfuns, *invars, **bind_params
                    )
                    # if eqn.primitive.multiple_results:
                    #     bad = any(bool(_has_nan(a)) for a in ans)
                    #     bad |= any(bool(_has_inf(a)) for a in ans)
                    # else:
                    #     bad = bool(_has_nan(ans))
                    #     bad |= bool(_has_inf(ans))

                    # if bad:
                    #     print("\n[NAN/inf DETECTED]")
                    #     print("primitive:", eqn.primitive, "id:", i)
                    #     print("invars:")
                    #     for k, v in enumerate(invars):
                    #         print(f"  arg{k}:", v)
                    #     print("out:", ans)
                    #     raise FloatingPointError(f"NaN/inf detected after primitive {eqn.primitive} (id: {i})")
                except KeyError:
                    raise NotImplementedError(
                        f"{eqn.primitive} not in inclusion_registry"
                    )
            else:
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)
    return safe_map(read, jaxpr.outvars)


def _make_inclusion_passthrough_p(primitive: Primitive) -> Callable[..., Interval]:
    """Creates an inclusion function that applies to the lower and upper bounds individually."""

    def _inclusion_p(*args, **kwargs) -> Interval:
        # Traverse args (possibly pytree) to get lower and upper bounds
        isinterval = lambda x: isinstance(x, Interval)
        getlower = lambda x: x.lower if isinstance(x, Interval) else x
        getupper = lambda x: x.upper if isinstance(x, Interval) else x
        args_l = jax.tree_util.tree_map(getlower, args, is_leaf=isinterval)
        args_u = jax.tree_util.tree_map(getupper, args, is_leaf=isinterval)
        return Interval(
            primitive.bind(*args_l, **kwargs), primitive.bind(*args_u, **kwargs)
        )

    return _inclusion_p


def _add_passthrough_to_registry(primitive: Primitive) -> None:
    """Helper to add a passthrough primitive to the inclusion registry."""
    inclusion_registry[primitive] = _make_inclusion_passthrough_p(primitive)


def _copy_predicate_metadata(dst: Interval, src: Interval) -> Interval:
    if hasattr(src, "_true_domain"):
        dst._true_domain = src._true_domain
    if hasattr(src, "_source_bounds"):
        dst._source_bounds = src._source_bounds
    return dst


class _DiscreteEnclosure:
    """Internal, finite-valued enclosure carried between Jaxpr primitives.

    ``Interval`` remains the public representation.  This metadata prevents a
    disconnected set of possible indices, such as ``{0, 9}``, from being
    prematurely widened to every integer in ``[0, 9]`` while the index is used
    by a later gather/dynamic-slice operation.
    """

    def __init__(self, lower, upper, mask, source):
        self.lower = lower
        self.upper = upper
        self.mask = mask
        self.source = source


def _discrete_enclosure(x):
    return getattr(x, "_discrete_enclosure", None) if isinstance(x, Interval) else None


def _masked_hull(lower, upper, mask):
    """Hull scenario bounds along their final (finite-choice) dimension."""
    out_lower = lower[..., 0]
    out_upper = upper[..., 0]
    seen = jnp.zeros_like(mask[..., 0], dtype=jnp.bool_)
    for i in range(lower.shape[-1]):
        active = mask[..., i]
        out_lower = jnp.where(
            active,
            jnp.where(seen, jnp.minimum(out_lower, lower[..., i]), lower[..., i]),
            out_lower,
        )
        out_upper = jnp.where(
            active,
            jnp.where(seen, jnp.maximum(out_upper, upper[..., i]), upper[..., i]),
            out_upper,
        )
        seen = jnp.logical_or(seen, active)
    return out_lower, out_upper


def _with_discrete_enclosure(
    out: Interval, lower, upper, mask, source, *, tighten=True
) -> Interval:
    if tighten:
        out.lower, out.upper = _masked_hull(lower, upper, mask)
    out._discrete_enclosure = _DiscreteEnclosure(lower, upper, mask, source)
    return out


def _scenario_bounds(x, source, scenario_count):
    """Lift an interval into scenarios, retaining correlation when available."""
    x = interval(x)
    discrete = _discrete_enclosure(x)
    if discrete is not None:
        if discrete.source is not source:
            return None
        return discrete.lower, discrete.upper, discrete.mask
    mask = jnp.ones(x.shape + (scenario_count,), dtype=jnp.bool_)
    return x.lower[..., None], x.upper[..., None], mask


def _inclusion_argminmax_p(x, *, axes, index_dtype, is_min):
    """Keep the exact feasible argmin/argmax indices as internal metadata."""
    x = interval(x)
    axis, = axes
    lower = jnp.moveaxis(x.lower, axis, -1)
    upper = jnp.moveaxis(x.upper, axis, -1)
    size = lower.shape[-1]
    reduction_axis = lower.ndim - 1

    if is_min:
        # JAX resolves equal extrema to the first index.  An earlier competitor
        # must therefore be made strictly larger than candidate i.
        earlier_extreme = lax.cummin(upper, axis=reduction_axis)
        later_extreme = lax.cummin(upper, axis=reduction_axis, reverse=True)
        earlier_feasible = jnp.concatenate(
            [
                jnp.ones_like(lower[..., :1], dtype=jnp.bool_),
                lower[..., 1:] < earlier_extreme[..., :-1],
            ],
            axis=-1,
        )
        later_feasible = jnp.concatenate(
            [
                lower[..., :-1] <= later_extreme[..., 1:],
                jnp.ones_like(lower[..., :1], dtype=jnp.bool_),
            ],
            axis=-1,
        )
    else:
        earlier_extreme = lax.cummax(lower, axis=reduction_axis)
        later_extreme = lax.cummax(lower, axis=reduction_axis, reverse=True)
        earlier_feasible = jnp.concatenate(
            [
                jnp.ones_like(upper[..., :1], dtype=jnp.bool_),
                upper[..., 1:] > earlier_extreme[..., :-1],
            ],
            axis=-1,
        )
        later_feasible = jnp.concatenate(
            [
                upper[..., :-1] >= later_extreme[..., 1:],
                jnp.ones_like(upper[..., :1], dtype=jnp.bool_),
            ],
            axis=-1,
        )

    candidate_mask = jnp.logical_and(earlier_feasible, later_feasible)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        # NaN endpoint semantics are not meaningful interval bounds.  Retain
        # soundness by considering every index possible for such a slice.
        has_nan = jnp.any(jnp.isnan(lower) | jnp.isnan(upper), axis=-1)
        candidate_mask = jnp.where(has_nan[..., None], True, candidate_mask)

    values = jnp.broadcast_to(
        jnp.arange(size, dtype=index_dtype), candidate_mask.shape
    )
    out_lower, out_upper = _masked_hull(values, values, candidate_mask)
    out = Interval(out_lower, out_upper)
    return _with_discrete_enclosure(
        out, values, values, candidate_mask, object(), tighten=False
    )


def _inclusion_argmin_p(x, **kwargs):
    return _inclusion_argminmax_p(x, is_min=True, **kwargs)


def _inclusion_argmax_p(x, **kwargs):
    return _inclusion_argminmax_p(x, is_min=False, **kwargs)


def _inclusion_broadcast_in_dim_p(x, **kwargs):
    out = _make_inclusion_passthrough_p(lax.broadcast_in_dim_p)(x, **kwargs)
    discrete = _discrete_enclosure(x)
    if discrete is None:
        return out

    shape = tuple(kwargs["shape"])
    dimensions = tuple(kwargs["broadcast_dimensions"])
    scenario_count = discrete.lower.shape[-1]
    scenario_kwargs = dict(kwargs)
    scenario_kwargs["shape"] = shape + (scenario_count,)
    scenario_kwargs["broadcast_dimensions"] = dimensions + (len(shape),)
    lower = lax.broadcast_in_dim_p.bind(discrete.lower, **scenario_kwargs)
    upper = lax.broadcast_in_dim_p.bind(discrete.upper, **scenario_kwargs)
    mask = lax.broadcast_in_dim_p.bind(discrete.mask, **scenario_kwargs)
    return _with_discrete_enclosure(
        out, lower, upper, mask, discrete.source, tighten=False
    )


def _inclusion_dynamic_slice_p(operand, *start_indices, **kwargs):
    """Hull slices over the finite index choices retained by argmin/argmax."""
    operand = interval(operand)
    uncertain = [
        (i, _discrete_enclosure(index))
        for i, index in enumerate(start_indices)
        if _discrete_enclosure(index) is not None
    ]
    if len(uncertain) != 1:
        return _make_inclusion_passthrough_p(lax.dynamic_slice_p)(
            operand, *start_indices, **kwargs
        )

    uncertain_position, discrete = uncertain[0]
    if discrete.lower.ndim != 1:
        raise NotImplementedError(
            "dynamic_slice currently supports scalar argmin/argmax indices"
        )

    scenario_lowers = []
    scenario_uppers = []
    for scenario in range(discrete.lower.shape[-1]):
        starts = [
            index.lower if isinstance(index, Interval) else index
            for index in start_indices
        ]
        starts[uncertain_position] = discrete.lower[scenario]
        scenario_lowers.append(
            lax.dynamic_slice_p.bind(operand.lower, *starts, **kwargs)
        )
        scenario_uppers.append(
            lax.dynamic_slice_p.bind(operand.upper, *starts, **kwargs)
        )

    lower = jnp.moveaxis(jnp.stack(scenario_lowers), 0, -1)
    upper = jnp.moveaxis(jnp.stack(scenario_uppers), 0, -1)
    mask = jnp.broadcast_to(discrete.mask, lower.shape)
    out = Interval(lower[..., 0], upper[..., 0])
    return _with_discrete_enclosure(out, lower, upper, mask, discrete.source)


def _inclusion_gather_p(operand, start_indices, **kwargs):
    """Hull gather results over retained finite index choices."""
    operand = interval(operand)
    discrete = _discrete_enclosure(start_indices)
    if discrete is None:
        return _make_inclusion_passthrough_p(lax.gather_p)(
            operand, start_indices, **kwargs
        )

    scenario_lowers = []
    scenario_uppers = []
    for scenario in range(discrete.lower.shape[-1]):
        indices = discrete.lower[..., scenario]
        scenario_lowers.append(lax.gather_p.bind(operand.lower, indices, **kwargs))
        scenario_uppers.append(lax.gather_p.bind(operand.upper, indices, **kwargs))

    lower = jnp.moveaxis(jnp.stack(scenario_lowers), 0, -1)
    upper = jnp.moveaxis(jnp.stack(scenario_uppers), 0, -1)
    # Gather output positions inherit the candidate mask of the corresponding
    # index position.  Broadcasting handles scalar and common advanced-index
    # forms; if JAX produces a more exotic layout, conservatively retain all
    # enumerated scenarios.
    try:
        # The penultimate dimension is gather's index-vector dimension; all
        # components of an index vector must belong to the same scenario.
        candidate_mask = (
            jnp.all(discrete.mask, axis=-2)
            if discrete.mask.ndim >= 2
            else discrete.mask
        )
        mask = jnp.broadcast_to(candidate_mask, lower.shape)
    except ValueError:
        mask = jnp.ones_like(lower, dtype=jnp.bool_)
    out = Interval(lower[..., 0], upper[..., 0])
    return _with_discrete_enclosure(out, lower, upper, mask, discrete.source)


# We would like to passthrough array operations like reshaping, slicing, etc.
_add_passthrough_to_registry(lax.copy_p)
_add_passthrough_to_registry(lax.reshape_p)
_add_passthrough_to_registry(lax.slice_p)
_add_passthrough_to_registry(lax.split_p)
_add_passthrough_to_registry(lax.squeeze_p)
_add_passthrough_to_registry(lax.transpose_p)
_add_passthrough_to_registry(lax.concatenate_p)
_add_passthrough_to_registry(lax.scatter_p)
_add_passthrough_to_registry(lax.scatter_add_p)
_add_passthrough_to_registry(lax.scatter_max_p)
_add_passthrough_to_registry(lax.scatter_min_p)
if hasattr(lax, "select_p"):
    _add_passthrough_to_registry(lax.select_p)
if hasattr(lax, "select_n_p"):
    _add_passthrough_to_registry(lax.select_n_p)
_add_passthrough_to_registry(lax.iota_p)
_add_passthrough_to_registry(lax.stop_gradient_p)
_add_passthrough_to_registry(lax.eq_p)
_add_passthrough_to_registry(lax.convert_element_type_p)
_add_passthrough_to_registry(lax.reduce_max_p)
_add_passthrough_to_registry(lax.reduce_min_p)
_add_passthrough_to_registry(lax.max_p)
_add_passthrough_to_registry(lax.min_p)
_add_passthrough_to_registry(lax.exp_p)
_add_passthrough_to_registry(lax.reduce_sum_p)
_add_passthrough_to_registry(lax.reduce_and_p)
_add_passthrough_to_registry(lax.reduce_or_p)
_add_passthrough_to_registry(lax.pad_p)
_add_passthrough_to_registry(lax.ne_p)
_add_passthrough_to_registry(lax.lt_p)
_add_passthrough_to_registry(lax.lt_to_p)
_add_passthrough_to_registry(lax.le_p)
_add_passthrough_to_registry(lax.gt_p)
_add_passthrough_to_registry(lax.ge_p)

_add_passthrough_to_registry(lax.and_p)
_add_passthrough_to_registry(lax.or_p)
_add_passthrough_to_registry(lax.not_p)

_add_passthrough_to_registry(debug_callback_p)

inclusion_registry[lax.argmin_p] = _inclusion_argmin_p
inclusion_registry[lax.argmax_p] = _inclusion_argmax_p
inclusion_registry[lax.broadcast_in_dim_p] = _inclusion_broadcast_in_dim_p
inclusion_registry[lax.dynamic_slice_p] = _inclusion_dynamic_slice_p
inclusion_registry[lax.gather_p] = _inclusion_gather_p

"""
TODO: Handle higher order primitives

natif_jaxpr should be thought of as an interpreter.
    - evaluates a jaxpr with interval arguments
    - uses the inclusion functions from the inclusion_registry
it cannot currently handle higher order primitives like scan, pjit
    - These HO primitives trace a jaxpr as their evaluation.
    - We can use natif_jaxpr to handle the jaxpr subexpression.
    - The inputs and outputs to the HO primitive itself are not correctly being handled.
Option 1:
    - make 'inclusion functions' which lowers intervals to pytrees and call the sub jaxpr with the proper conversions.
    - downside: this would be needed for each HO primitive.
Option 2:
    - handle them in a more principled manner, maybe during the tracing step
    - when we trace, we can also extract from the pytree which nodes will be intervals
    - somehow, if we see a HO primitive, perhaps changing the inputs will suffice
    - downside: requires more work to understand how to do this.

In principle, the only problem is the inputs to the HO primitive are not jax types
Natively, we cannot pass in pytrees like we are trying here.
"""

# Some higher order primitives


def _inclusion_pjit_p(*args, **bind_params) -> Interval:
    """For now, this ignores a pjit_p and returns the evaluation of the jaxpr."""
    # TODO: Do we need to implement consts here?
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    return natif_jaxpr(bind_jaxpr, [], *args)


# def _inclusion_pjit_p(*args, **bind_params):
#     bind_jaxpr = bind_params["jaxpr"]
#     if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
#         return natif_jaxpr(bind_jaxpr.jaxpr, bind_jaxpr.consts, *args)
#     return natif_jaxpr(bind_jaxpr, [], *args)

inclusion_registry[jax._src.pjit.jit_p] = _inclusion_pjit_p


def _inclusion_scan_p(*args, **bind_params):
    bind_jaxpr = bind_params["jaxpr"]
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        consts = list(bind_jaxpr.consts)
        bind_jaxpr = bind_jaxpr.jaxpr
    else:
        consts = []

    num_consts = bind_params["num_consts"]
    num_carry = bind_params["num_carry"]
    length = bind_params["length"]
    reverse = bind_params.get("reverse", False)

    # Flat argument structure for scan_p:
    #   args = consts_from_args + carry + xs
    scan_consts = list(args[:num_consts])
    carry = list(args[num_consts:num_consts + num_carry])
    xs = list(args[num_consts + num_carry:])

    # If jaxpr came as ClosedJaxpr, prefer its embedded consts unless empty.
    if len(consts) == 0:
        consts = scan_consts

    def _slice_x(x, i):
        if isinstance(x, Interval):
            return Interval(x.lower[i], x.upper[i])
        else:
            return x[i]

    def _stack_seq(seq):
        # seq is a list over time of values, where each value may be Interval or array
        first = seq[0]
        if isinstance(first, Interval):
            return Interval(
                jnp.stack([v.lower for v in seq], axis=0),
                jnp.stack([v.upper for v in seq], axis=0),
            )
        else:
            return jnp.stack(seq, axis=0)

    ys_over_time = None

    indices = range(length - 1, -1, -1) if reverse else range(length)

    for i in indices:
        x_t = [_slice_x(x, i) for x in xs]
        invals = [*consts, *carry, *x_t]

        outvals = list(natif_jaxpr(bind_jaxpr, [], *invals))

        new_carry = outvals[:num_carry]
        y_t = outvals[num_carry:]

        carry = new_carry

        if ys_over_time is None:
            ys_over_time = [[] for _ in range(len(y_t))]
        for k, yk in enumerate(y_t):
            ys_over_time[k].append(yk)

    if ys_over_time is None:
        ys = []
    else:
        if reverse:
            ys_over_time = [list(reversed(vs)) for vs in ys_over_time]
        ys = [_stack_seq(vs) for vs in ys_over_time]

    return [*carry, *ys]

inclusion_registry[lax.scan_p] = _inclusion_scan_p


def _inclusion_while_p(*args, **bind_params):
    cond_jaxpr = bind_params["cond_jaxpr"]
    body_jaxpr = bind_params["body_jaxpr"]
    if isinstance(cond_jaxpr, jax.extend.core.ClosedJaxpr):
        cond_consts = list(cond_jaxpr.consts)
        cond_jaxpr = cond_jaxpr.jaxpr
    else:
        cond_consts = []
    if isinstance(body_jaxpr, jax.extend.core.ClosedJaxpr):
        body_consts = list(body_jaxpr.consts)
        body_jaxpr = body_jaxpr.jaxpr
    else:
        body_consts = []

    cond_nconsts = bind_params["cond_nconsts"]
    body_nconsts = bind_params["body_nconsts"]
    cond_consts_from_args = list(args[:cond_nconsts])
    body_consts_from_args = list(args[cond_nconsts:cond_nconsts + body_nconsts])
    carry = list(args[cond_nconsts + body_nconsts:])

    if len(cond_consts) == 0:
        cond_consts = cond_consts_from_args
    if len(body_consts) == 0:
        body_consts = body_consts_from_args
    carry = [x if isinstance(x, Interval) else interval(x) for x in carry]

    def cond_fun(carry):
        out = natif_jaxpr(cond_jaxpr, [], *cond_consts, *carry)[0]
        return out.lower if isinstance(out, Interval) else out

    def body_fun(carry):
        return tuple(x if isinstance(x, Interval) else interval(x)
                     for x in natif_jaxpr(body_jaxpr, [], *body_consts, *carry))

    return list(lax.while_loop(cond_fun, body_fun, tuple(carry)))


inclusion_registry[lax.while_p] = _inclusion_while_p


def _inclusion_cond_p(index, *args, **bind_params):
    branches = bind_params["branches"]
    index = index.lower if isinstance(index, Interval) else index

    branch_outs = []
    for branch in branches:
        if isinstance(branch, jax.extend.core.ClosedJaxpr):
            consts = list(branch.consts)
            branch_jaxpr = branch.jaxpr
        else:
            consts = []
            branch_jaxpr = branch
        branch_outs.append(tuple(natif_jaxpr(branch_jaxpr, [], *consts, *args)))

    def _select(*xs):
        if any(isinstance(x, Interval) for x in xs):
            xs = [interval(x) for x in xs]

            def _lower(x):
                while isinstance(x, Interval):
                    x = x.lower
                return x

            def _upper(x):
                while isinstance(x, Interval):
                    x = x.upper
                return x

            return Interval(
                lax.select_n(index, *[_lower(x) for x in xs]),
                lax.select_n(index, *[_upper(x) for x in xs]),
            )
        return lax.select_n(index, *xs)

    return [
        jax.tree_util.tree_map(_select, *out_group, is_leaf=lambda x: isinstance(x, Interval))
        for out_group in zip(*branch_outs)
    ]


inclusion_registry[lax.cond_p] = _inclusion_cond_p


def _inclusion_add_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        out = Interval(x.lower + y.lower, x.upper + y.upper)
    elif isinstance(x, Interval):
        out = Interval(x.lower + y, x.upper + y)
    elif isinstance(y, Interval):
        out = Interval(x + y.lower, x + y.upper)
    else:
        return x + y

    x_discrete = _discrete_enclosure(x)
    y_discrete = _discrete_enclosure(y)
    discrete = x_discrete or y_discrete
    if discrete is None:
        return out
    if (
        x_discrete is not None
        and y_discrete is not None
        and x_discrete.source is not y_discrete.source
    ):
        return out

    scenario_count = discrete.lower.shape[-1]
    x_scenarios = _scenario_bounds(x, discrete.source, scenario_count)
    y_scenarios = _scenario_bounds(y, discrete.source, scenario_count)
    if x_scenarios is None or y_scenarios is None:
        return out
    xl, xu, xm = x_scenarios
    yl, yu, ym = y_scenarios
    return _with_discrete_enclosure(
        out,
        xl + yl,
        xu + yu,
        jnp.logical_and(xm, ym),
        discrete.source,
    )


inclusion_registry[lax.add_p] = _inclusion_add_p
inclusion_registry[ad_util.add_any_p] = _inclusion_add_p
Interval.__add__ = _inclusion_add_p


def _inclusion_sub_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(x.lower - y.upper, x.upper - y.lower)
    elif isinstance(x, Interval):
        return Interval(x.lower - y, x.upper - y)
    elif isinstance(y, Interval):
        return Interval(x - y.upper, x - y.lower)
    else:
        return x - y


inclusion_registry[lax.sub_p] = _inclusion_sub_p
Interval.__sub__ = _inclusion_sub_p


def _inclusion_neg_p(x: Interval) -> Interval:
    return Interval(-x.upper, -x.lower)


inclusion_registry[lax.neg_p] = _inclusion_neg_p
Interval.__neg__ = _inclusion_neg_p


def _inclusion_mul_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        x_domain = getattr(y, "_true_domain", None)
        if x_domain is not None and getattr(y, "_source_bounds", None) == (id(x.lower), id(x.upper)):
            return _hull2(Interval(jnp.zeros_like(x_domain.lower), jnp.zeros_like(x_domain.upper)), x_domain)

        y_domain = getattr(x, "_true_domain", None)
        if y_domain is not None and getattr(x, "_source_bounds", None) == (id(y.lower), id(y.upper)):
            return _hull2(Interval(jnp.zeros_like(y_domain.lower), jnp.zeros_like(y_domain.upper)), y_domain)

    if isinstance(x, Interval) and isinstance(y, Interval):
        _1 = x.lower * y.lower
        _2 = x.lower * y.upper
        _3 = x.upper * y.lower
        _4 = x.upper * y.upper
        return Interval(
            jnp.minimum(jnp.minimum(_1, _2), jnp.minimum(_3, _4)),
            jnp.maximum(jnp.maximum(_1, _2), jnp.maximum(_3, _4)),
        )
    elif isinstance(x, Interval):
        _1 = x.lower * y
        _2 = x.upper * y
        return Interval(jnp.minimum(_1, _2), jnp.maximum(_1, _2))
    elif isinstance(y, Interval):
        _1 = x * y.lower
        _2 = x * y.upper
        return Interval(jnp.minimum(_1, _2), jnp.maximum(_1, _2))
    else:
        return x * y


inclusion_registry[lax.mul_p] = _inclusion_mul_p
Interval.__mul__ = _inclusion_mul_p


def _comparison_bounds(x, y):
    x = interval(x) if isinstance(x, Interval) else x
    y = interval(y) if isinstance(y, Interval) else y
    xl = x.lower if isinstance(x, Interval) else x
    xu = x.upper if isinstance(x, Interval) else x
    yl = y.lower if isinstance(y, Interval) else y
    yu = y.upper if isinstance(y, Interval) else y
    return xl, xu, yl, yu


def _with_true_domain(pred: Interval, source: Interval, lower, upper) -> Interval:
    pred._true_domain = Interval(jnp.maximum(source.lower, lower), jnp.minimum(source.upper, upper))
    pred._source_bounds = (id(source.lower), id(source.upper))
    return pred


def _with_discrete_comparison(pred, x, y, comparison):
    x_discrete = _discrete_enclosure(x)
    y_discrete = _discrete_enclosure(y)
    discrete = x_discrete or y_discrete
    if discrete is None:
        return pred
    if (
        x_discrete is not None
        and y_discrete is not None
        and x_discrete.source is not y_discrete.source
    ):
        return pred

    scenario_count = discrete.lower.shape[-1]
    x_scenarios = _scenario_bounds(x, discrete.source, scenario_count)
    y_scenarios = _scenario_bounds(y, discrete.source, scenario_count)
    if x_scenarios is None or y_scenarios is None:
        return pred
    xl, xu, xm = x_scenarios
    yl, yu, ym = y_scenarios

    if comparison == "lt":
        lower, upper = xu < yl, xl < yu
    elif comparison == "le":
        lower, upper = xu <= yl, xl <= yu
    elif comparison == "gt":
        lower, upper = xl > yu, xu > yl
    elif comparison == "ge":
        lower, upper = xl >= yu, xu >= yl
    else:
        raise ValueError(f"Unknown comparison {comparison}")
    return _with_discrete_enclosure(
        pred,
        lower,
        upper,
        jnp.logical_and(xm, ym),
        discrete.source,
    )


def _inclusion_lt_p(x, y) -> Interval:
    xl, xu, yl, yu = _comparison_bounds(x, y)
    pred = Interval(xu < yl, xl < yu)
    pred = _with_discrete_comparison(pred, x, y, "lt")
    if isinstance(x, Interval) and not isinstance(y, Interval):
        pred = _with_true_domain(pred, x, -jnp.inf, y)
    return pred


def _inclusion_le_p(x, y) -> Interval:
    xl, xu, yl, yu = _comparison_bounds(x, y)
    pred = Interval(xu <= yl, xl <= yu)
    pred = _with_discrete_comparison(pred, x, y, "le")
    if isinstance(x, Interval) and not isinstance(y, Interval):
        pred = _with_true_domain(pred, x, -jnp.inf, y)
    return pred


def _inclusion_gt_p(x, y) -> Interval:
    xl, xu, yl, yu = _comparison_bounds(x, y)
    pred = Interval(xl > yu, xu > yl)
    pred = _with_discrete_comparison(pred, x, y, "gt")
    if isinstance(x, Interval) and not isinstance(y, Interval):
        pred = _with_true_domain(pred, x, y, jnp.inf)
    return pred


def _inclusion_ge_p(x, y) -> Interval:
    xl, xu, yl, yu = _comparison_bounds(x, y)
    pred = Interval(xl >= yu, xu >= yl)
    pred = _with_discrete_comparison(pred, x, y, "ge")
    if isinstance(x, Interval) and not isinstance(y, Interval):
        pred = _with_true_domain(pred, x, y, jnp.inf)
    return pred


def _inclusion_eq_p(x, y) -> Interval:
    xl, xu, yl, yu = _comparison_bounds(x, y)
    definitely = (xl == xu) & (yl == yu) & (xl == yl)
    possibly = (xl <= yu) & (yl <= xu)
    return Interval(definitely, possibly)


def _inclusion_ne_p(x, y) -> Interval:
    eq = _inclusion_eq_p(x, y)
    return Interval(jnp.logical_not(eq.upper), jnp.logical_not(eq.lower))


def _inclusion_and_p(x, y) -> Interval:
    x = interval(x) if isinstance(x, Interval) else Interval(x, x)
    y = interval(y) if isinstance(y, Interval) else Interval(y, y)
    out = Interval(jnp.logical_and(x.lower, y.lower), jnp.logical_and(x.upper, y.upper))
    x_source = getattr(x, "_source_bounds", None)
    if x_source is not None and x_source == getattr(y, "_source_bounds", None):
        xd = getattr(x, "_true_domain", None)
        yd = getattr(y, "_true_domain", None)
        if xd is not None and yd is not None:
            out._true_domain = Interval(jnp.maximum(xd.lower, yd.lower), jnp.minimum(xd.upper, yd.upper))
            out._source_bounds = x_source
    return out


def _inclusion_or_p(x, y) -> Interval:
    x = interval(x) if isinstance(x, Interval) else Interval(x, x)
    y = interval(y) if isinstance(y, Interval) else Interval(y, y)
    return Interval(jnp.logical_or(x.lower, y.lower), jnp.logical_or(x.upper, y.upper))


def _inclusion_not_p(x) -> Interval:
    x = interval(x) if isinstance(x, Interval) else Interval(x, x)
    return Interval(jnp.logical_not(x.upper), jnp.logical_not(x.lower))


def _inclusion_convert_element_type_p(x, **kwargs) -> Interval:
    out = _make_inclusion_passthrough_p(lax.convert_element_type_p)(x, **kwargs)
    if isinstance(x, Interval):
        out = _copy_predicate_metadata(out, x)
    return out


def _hull2(x, y) -> Interval:
    x = interval(x)
    y = interval(y)
    return Interval(jnp.minimum(x.lower, y.lower), jnp.maximum(x.upper, y.upper))


def _inclusion_select_p(pred, on_true, on_false) -> Interval:
    pred = interval(pred) if isinstance(pred, Interval) else Interval(pred, pred)
    on_true = interval(on_true)
    on_false = interval(on_false)
    true_only = pred.lower
    false_only = jnp.logical_not(pred.upper)
    hull = _hull2(on_true, on_false)
    return Interval(
        jnp.where(true_only, on_true.lower, jnp.where(false_only, on_false.lower, hull.lower)),
        jnp.where(true_only, on_true.upper, jnp.where(false_only, on_false.upper, hull.upper)),
    )


def _inclusion_select_n_p(which, *cases) -> Interval:
    which = interval(which) if isinstance(which, Interval) else Interval(which, which)
    cases = [interval(c) for c in cases]
    if len(cases) == 2 and which.lower.dtype == jnp.bool_:
        false_case, true_case = cases
        true_only = which.lower
        false_only = jnp.logical_not(which.upper)
        hull = _hull2(false_case, true_case)
        out = Interval(
            jnp.where(true_only, true_case.lower, jnp.where(false_only, false_case.lower, hull.lower)),
            jnp.where(true_only, true_case.upper, jnp.where(false_only, false_case.upper, hull.upper)),
        )
        discrete_parts = [
            discrete
            for discrete in (
                _discrete_enclosure(which),
                _discrete_enclosure(false_case),
                _discrete_enclosure(true_case),
            )
            if discrete is not None
        ]
        if not discrete_parts:
            return out
        source = discrete_parts[0].source
        if any(discrete.source is not source for discrete in discrete_parts[1:]):
            return out

        scenario_count = discrete_parts[0].lower.shape[-1]
        which_scenarios = _scenario_bounds(which, source, scenario_count)
        false_scenarios = _scenario_bounds(false_case, source, scenario_count)
        true_scenarios = _scenario_bounds(true_case, source, scenario_count)
        if any(
            scenarios is None
            for scenarios in (which_scenarios, false_scenarios, true_scenarios)
        ):
            return out
        wl, wu, wm = which_scenarios
        fl, fu, fm = false_scenarios
        tl, tu, tm = true_scenarios
        scenario_hull_lower = jnp.minimum(fl, tl)
        scenario_hull_upper = jnp.maximum(fu, tu)
        lower = jnp.where(wl, tl, jnp.where(jnp.logical_not(wu), fl, scenario_hull_lower))
        upper = jnp.where(wl, tu, jnp.where(jnp.logical_not(wu), fu, scenario_hull_upper))
        mask = jnp.logical_and(wm, jnp.logical_and(fm, tm))
        return _with_discrete_enclosure(out, lower, upper, mask, source)
    lower = jnp.full_like(cases[0].lower, jnp.inf)
    upper = jnp.full_like(cases[0].upper, -jnp.inf)
    for i, case in enumerate(cases):
        maybe_i = jnp.logical_and(which.lower <= i, i <= which.upper)
        lower = jnp.where(maybe_i, jnp.minimum(lower, case.lower), lower)
        upper = jnp.where(maybe_i, jnp.maximum(upper, case.upper), upper)
    return Interval(lower, upper)


inclusion_registry[lax.lt_p] = _inclusion_lt_p
if hasattr(lax, "lt_to_p"):
    inclusion_registry[lax.lt_to_p] = _inclusion_lt_p
inclusion_registry[lax.le_p] = _inclusion_le_p
inclusion_registry[lax.gt_p] = _inclusion_gt_p
inclusion_registry[lax.ge_p] = _inclusion_ge_p
inclusion_registry[lax.eq_p] = _inclusion_eq_p
inclusion_registry[lax.convert_element_type_p] = _inclusion_convert_element_type_p
inclusion_registry[lax.ne_p] = _inclusion_ne_p
inclusion_registry[lax.and_p] = _inclusion_and_p
inclusion_registry[lax.or_p] = _inclusion_or_p
inclusion_registry[lax.not_p] = _inclusion_not_p
if hasattr(lax, "select_p"):
    inclusion_registry[lax.select_p] = _inclusion_select_p
if hasattr(lax, "select_n_p"):
    inclusion_registry[lax.select_n_p] = _inclusion_select_n_p


def _inclusion_div_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    elif isinstance(x, Interval):
        return _inclusion_mul_p(x, 1 / y)
    elif isinstance(y, Interval):
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    else:
        return x / y


inclusion_registry[lax.div_p] = _inclusion_div_p
Interval.__truediv__ = _inclusion_div_p


def _inclusion_reciprocal_p(x: Interval) -> Interval:
    if not isinstance(x, Interval):
        return 1 / x
    c = jnp.logical_or(
        jnp.logical_and(x.lower > 0, x.upper > 0),
        jnp.logical_and(x.lower < 0, x.upper < 0),
    )
    return Interval(
        jnp.where(c, (1.0 / x.upper), -jnp.inf), jnp.where(c, (1.0 / x.lower), jnp.inf)
    )


def _inclusion_integer_pow_p(x: Interval, y: int) -> Interval:
    if not isinstance(x, Interval):
        return x**y

    def _inclusion_integer_pow_impl(x: Interval, y: int) -> Interval:
        l_pow = lax.integer_pow(x.lower, y)
        u_pow = lax.integer_pow(x.upper, y)

        def even():
            contains_zero = jnp.logical_and(
                jnp.less_equal(x.lower, 0), jnp.greater_equal(x.upper, 0)
            )
            lower = jnp.where(
                contains_zero, jnp.zeros_like(x.lower), jnp.minimum(l_pow, u_pow)
            )
            upper = jnp.maximum(l_pow, u_pow)
            return (lower, upper)

        odd = lambda: (l_pow, u_pow)

        return lax.cond(jnp.all(y % 2), odd, even)

    def _pos_pow():
        return _inclusion_integer_pow_impl(x, y)

    def _neg_pow():
        return _inclusion_integer_pow_impl(_inclusion_reciprocal_p(x), -y)

    ol, ou = lax.cond(jnp.all(y < 0), _neg_pow, _pos_pow)
    return Interval(ol, ou)


inclusion_registry[lax.integer_pow_p] = _inclusion_integer_pow_p
Interval.__pow__ = _inclusion_integer_pow_p


def _inclusion_square_p(x: Interval) -> Interval:
    """Square an interval."""
    return _inclusion_integer_pow_p(x, 2)


inclusion_registry[lax.square_p] = _inclusion_square_p


def _inclusion_dot_general_p(A: Interval, B: Interval, **kwargs) -> Interval:
    # All checks of batch/contracting dims are done in first pass on lower bounds

    A = interval(A)
    B = interval(B)

    # Extract the contracting and batch dimensions
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = kwargs[
        "dimension_numbers"
    ]

    # Permute the batch then contracting dimensions to the front
    imoveaxis = lambda x, *args: Interval(
        jnp.moveaxis(x.lower, *args), jnp.moveaxis(x.upper, *args)
    )
    A = imoveaxis(
        A, lhs_batch + lhs_contracting, range(len(lhs_batch) + len(lhs_contracting))
    )
    B = imoveaxis(
        B, rhs_batch + rhs_contracting, range(len(rhs_batch) + len(rhs_contracting))
    )

    def _contract(A, B):
        # Multiplying two scalar intervals
        def _mul(a, b):
            _1 = a.lower * b.lower
            _2 = a.lower * b.upper
            _3 = a.upper * b.lower
            _4 = a.upper * b.upper
            return Interval(
                jnp.minimum(jnp.minimum(_1, _2), jnp.minimum(_3, _4)),
                jnp.maximum(jnp.maximum(_1, _2), jnp.maximum(_3, _4)),
            )

        isum = lambda x: Interval(jnp.sum(x.lower), jnp.sum(x.upper))

        if len(lhs_contracting) == 0:
            a = A
            b = B
            for _ in range(B.lower.ndim):
                a = Interval(jnp.expand_dims(a.lower, -1), jnp.expand_dims(a.upper, -1))
            for _ in range(A.lower.ndim):
                b = Interval(jnp.expand_dims(b.lower, 0), jnp.expand_dims(b.upper, 0))
            return _mul(a, b)

        # Two vectors -> scalar
        def f(a, b):
            # _mulres = jax.vmap(_mul)(a, b)
            # return Interval(jnp.sum(_mulres.lower), jnp.sum(_mulres.upper))
            _r = jax.vmap(_mul)
            return isum(_r(a, b))

        # Repeat over each contracting dimension
        for i in range(1, len(lhs_contracting)):
            _r = jax.vmap(f)
            f = lambda a, b: isum(_r(a, b))

        # vmap over non-contracting dimensions
        for i in range(len(lhs_contracting), len(A.shape)):
            f = jax.vmap(f, in_axes=(i, None), out_axes=-1)
        for j in range(len(rhs_contracting), len(B.shape)):
            f = jax.vmap(f, in_axes=(None, j), out_axes=-1)

        return f(A, B)

    # vmap over batch dimensions
    f = _contract
    for i in range(len(lhs_batch)):
        f = vmap(f, in_axes=(0, 0), out_axes=0)

    return f(A, B)


inclusion_registry[lax.dot_general_p] = _inclusion_dot_general_p


def _inclusion_sin_p(x: Interval, accuracy=None) -> Interval:
    if not isinstance(x, Interval):
        return lax.sin(x, accuracy=accuracy)

    def _sin_if(l: jnp.float32, u: jnp.float32):
        def case_lpi(l, u):
            cl = jnp.cos(l)
            cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2 * jnp.array(cu >= 0, "int32")
            case3 = lambda: (jnp.sin(l), jnp.sin(u))  # cl >= 0, cu >= 0
            case0 = lambda: (jnp.sin(u), jnp.sin(l))  # cl <= 0, cu <= 0
            case1 = lambda: (
                jnp.minimum(jnp.sin(l), jnp.sin(u)),
                1.0,
            )  # cl >= 0, cu <= 0
            case2 = lambda: (
                -1.0,
                jnp.maximum(jnp.sin(l), jnp.sin(u)),
            )  # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])

        def case_pi2pi(l, u):
            cl = jnp.cos(l)
            cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2 * jnp.array(cu >= 0, "int32")
            case3 = lambda: (-1.0, 1.0)  # cl >= 0, cu >= 0
            case0 = lambda: (-1.0, 1.0)  # cl <= 0, cu <= 0
            case1 = lambda: (
                jnp.minimum(jnp.sin(l), jnp.sin(u)),
                1.0,
            )  # cl >= 0, cu <= 0
            case2 = lambda: (
                -1.0,
                jnp.maximum(jnp.sin(l), jnp.sin(u)),
            )  # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])

        def case_else(l, u):
            return -1.0, 1.0

        diff = u - l
        c = jnp.array(diff <= jnp.pi, "int32") + jnp.array(diff <= 2 * jnp.pi, "int32")
        ol, ou = lax.switch(c, [case_else, case_pi2pi, case_lpi], l, u)
        return ol, ou

    _sin_if_vmap = jax.vmap(_sin_if, (0, 0))
    _x, x_ = _sin_if_vmap(x.lower.reshape(-1), x.upper.reshape(-1))
    return Interval(_x.reshape(x.shape), x_.reshape(x.shape))


inclusion_registry[lax.sin_p] = _inclusion_sin_p


def _inclusion_cos_p(x: Interval, accuracy=None) -> Interval:
    return _inclusion_sin_p(
        Interval(x.lower + jnp.pi / 2, x.upper + jnp.pi / 2), accuracy=accuracy
    )


inclusion_registry[lax.cos_p] = _inclusion_cos_p


def _inclusion_tan_p(x: Interval, accuracy=None) -> Interval:
    l = x.lower
    u = x.upper
    div = jnp.floor((u + jnp.pi / 2) / (jnp.pi)).astype(int)
    l -= div * jnp.pi
    u -= div * jnp.pi
    ol = jnp.where((l < -jnp.pi / 2), -jnp.inf, jnp.tan(l))
    ou = jnp.where((l < -jnp.pi / 2), jnp.inf, jnp.tan(u))
    return Interval(ol, ou)


inclusion_registry[lax.tan_p] = _inclusion_tan_p

# def _inclusion_atan_p (x:Interval, accuracy=None) -> Interval :
#     return Interval(lax.atan(x.lower), lax.atan(x.upper))
# inclusion_registry[lax.atan_p] = _inclusion_atan_p
_add_passthrough_to_registry(lax.atan_p)


def _inclusion_asin_p(x: Interval, accuracy=None) -> Interval:
    return Interval(
        lax.arcsin(x.lower, accuracy=accuracy), lax.arcsin(x.upper, accuracy=accuracy)
    )


inclusion_registry[lax.asin_p] = _inclusion_asin_p


def _inclusion_sqrt_p(x: Interval, accuracy=None) -> Interval:
    ol = jnp.where((x.lower < 0), -jnp.inf, jnp.sqrt(x.lower))
    ou = jnp.where((x.lower < 0), -jnp.inf, jnp.sqrt(x.upper))
    # ol = jnp.where((x.lower < 0), 0, jnp.sqrt(x.lower))
    # ou = jnp.where((x.lower < 0), 0, jnp.sqrt(x.upper))
    return Interval(ol, ou)


inclusion_registry[lax.sqrt_p] = _inclusion_sqrt_p


def _inclusion_pow_p(x: Interval, y: Interval) -> Interval:
    # if isinstance (y, Interval) :
    #     # if y.lower == y.upper :
    #     if True :
    #         y = y.upper
    #     else :
    #         raise Exception('y must be a constant')

    x = interval(x)
    y = interval(y)

    def _inclusion_pow_impl(xl, xu, yl, yu) -> Interval:
        # caluclate corners
        corners = jnp.array(
            [lax.pow(xl, yl), lax.pow(xl, yu), lax.pow(xu, yl), lax.pow(xu, yu)]
        )
        # calculate the minimum and maximum of the corners
        cond = jnp.logical_and(xl >= 0, xu >= 0)
        ol = jnp.where(cond, jnp.min(corners), -jnp.inf)
        ou = jnp.where(cond, jnp.max(corners), jnp.inf)
        return ol, ou

    xl, xu, yl, yu = jnp.broadcast_arrays(x.lower, x.upper, y.lower, y.upper)
    xsh = jnp.shape(xl)

    resl, resu = jax.vmap(_inclusion_pow_impl, (0, 0, 0, 0))(
        xl.reshape(-1), xu.reshape(-1), yl.reshape(-1), yu.reshape(-1)
    )
    return Interval(resl.reshape(xsh), resu.reshape(xsh))


inclusion_registry[lax.pow_p] = _inclusion_pow_p


# def _inclusion_tanh_p(x: Interval, accuracy=None) -> Interval:
#     return Interval(lax.tanh(x.lower, accuracy=accuracy), lax.tanh(x.upper, accuracy=accuracy))

# inclusion_registry[lax.tanh_p] = _inclusion_tanh_p
_add_passthrough_to_registry(lax.tanh_p)

def _inclusion_log_p(x: Interval, accuracy=None) -> Interval :
    ol = jnp.where((x.lower < 0), -jnp.inf, jnp.log(x.lower))
    ou = jnp.where((x.lower < 0), -jnp.inf, jnp.log(x.upper))
    return Interval(ol, ou)

inclusion_registry[lax.log_p] = _inclusion_log_p

def _inclusion_log1p_p(x: Interval, accuracy=None) -> Interval :
    ol = jnp.where((x.lower < -1), -jnp.inf, jnp.log1p(x.lower))
    ou = jnp.where((x.lower < -1), -jnp.inf, jnp.log1p(x.upper))
    return Interval(ol, ou)

inclusion_registry[lax.log1p_p] = _inclusion_log1p_p

Interval.__matmul__ = jit(natif(jnp.matmul))

# Some linear algebra routines


# Cholesky decomposition
def _manual_cholesky(A):
    """
    Computes the Cholesky decomposition of a symmetric positive definite matrix A using Python for loops.
    Returns lower-triangular matrix L such that A = L @ L.T
    """
    A = 0.5 * (A + A.T)  # Ensure symmetry
    n = A.shape[0]
    L = jnp.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = jnp.sum(L[i, :j] * L[j, :j])
            val = jnp.where(i == j, jnp.sqrt(A[i, i] - s), (A[i, j] - s) / L[j, j])
            # if i == j:
            #     val = jnp.sqrt(A[i, i] - s)
            # else:
            #     val = (A[i, j] - s) / L[j, j]
            L = L.at[i, j].set(val)
    return L

def _manual_cholesky_masked_scan(A):
    A = 0.5 * (A + A.T)  # Ensure symmetry
    n = A.shape[0]
    idx = jnp.arange(n)
    L0 = jnp.zeros_like(A)

    def body(L, j):
        active_prev = idx < j
        active_col = idx == j
        active_rows = idx >= j

        a_col = jnp.sum(A * active_col[None, :], axis=1)
        row_j = jnp.sum(L * active_col[:, None], axis=0) * active_prev
        s = L @ row_j
        ajj = jnp.sum(a_col * active_col)
        sjj = jnp.sum(s * active_col)
        ljj = jnp.sqrt(ajj - sjj)

        raw_col = (a_col - s) / ljj
        new_col = jnp.where(active_col, ljj, raw_col)
        update_col = active_rows[:, None] & active_col[None, :]
        return jnp.where(update_col, new_col[:, None], L), None

    return lax.scan(body, L0, idx)[0]


inclusion_registry[LA.cholesky_p] = natif(_manual_cholesky_masked_scan)

# Triangular solve


def _manual_triangular_solve(
    A,
    b,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    if transpose_a:
        A = A.T
        lower = not lower
    if conjugate_a:
        A = jnp.conj(A)

    if not left_side:
        return _manual_triangular_solve(
            A.T,
            b.T,
            left_side=True,
            lower=not lower,
            transpose_a=False,
            conjugate_a=False,
            unit_diagonal=unit_diagonal,
        ).T

    squeeze_output = (b.ndim == 1)
    if squeeze_output:
        b = b[:, None]

    n = A.shape[0]
    x = jnp.zeros_like(b)

    def lower_body(i, x):
        s = A[i, :] @ x
        diag = 1.0 if unit_diagonal else A[i, i]
        xi = (b[i, :] - s) / diag
        return x.at[i, :].set(xi)

    def upper_body(i_rev, x):
        i = n - 1 - i_rev
        s = A[i, :] @ x
        diag = 1.0 if unit_diagonal else A[i, i]
        xi = (b[i, :] - s) / diag
        return x.at[i, :].set(xi)

    if lower:
        x = jax.lax.fori_loop(0, n, lower_body, x)
    else:
        x = jax.lax.fori_loop(0, n, upper_body, x)

    return x[:, 0] if squeeze_output else x


@partial(
    jit,
    static_argnames=(
        "left_side",
        "lower",
        "transpose_a",
        "conjugate_a",
        "unit_diagonal",
    ),
)
def _inclusion_triangular_solve(
    A,
    b,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    # return natif(partial(jax.vmap(_manual_triangular_solve, in_axes=()),
    #                      left_side=left_side, lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a, unit_diagonal=unit_diagonal))(A, b)
    return natif(
        partial(
            _manual_triangular_solve,
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=conjugate_a,
            unit_diagonal=unit_diagonal,
        )
    )(A, b)


inclusion_registry[LA.triangular_solve_p] = _inclusion_triangular_solve

# natif(lambda A, b, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False: _manual_triangular_solve(A, b, left_side=left_side, lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a, unit_diagonal=unit_diagonal))



def _inclusion_abs_p(x: Interval) -> Interval:
    l = x.lower
    u = x.upper

    # cases
    lower = jnp.where(
        l >= 0,
        l,                        # entirely positive
        jnp.where(
            u <= 0,
            -u,                   # entirely negative
            0.0                   # crosses zero
        )
    )

    upper = jnp.maximum(jnp.abs(l), jnp.abs(u))

    return Interval(lower, upper)


inclusion_registry[lax.abs_p] = _inclusion_abs_p
Interval.__abs__ = _inclusion_abs_p


def _inclusion_atan2_p(y: Interval, x: Interval) -> Interval:
    y = interval(y)
    x = interval(x)

    yl, yu, xl, xu = jnp.broadcast_arrays(y.lower, y.upper, x.lower, x.upper)
    out_shape = yl.shape

    def _atan2_elem(yl, yu, xl, xu):
        corners = jnp.array([
            lax.atan2(yl, xl),
            lax.atan2(yl, xu),
            lax.atan2(yu, xl),
            lax.atan2(yu, xu),
        ])

        # If the x-interval contains 0, atan2 can jump across quadrants / branch cut.
        # Use a conservative enclosure.
        x_crosses_zero = jnp.logical_and(xl <= 0, xu >= 0)

        # Also be conservative if both x and y intervals contain 0.
        y_crosses_zero = jnp.logical_and(yl <= 0, yu >= 0)
        ambiguous = jnp.logical_and(x_crosses_zero, y_crosses_zero)

        lower = jnp.where(
            jnp.logical_or(x_crosses_zero, ambiguous),
            -jnp.pi,
            jnp.min(corners),
        )
        upper = jnp.where(
            jnp.logical_or(x_crosses_zero, ambiguous),
            jnp.pi,
            jnp.max(corners),
        )
        return lower, upper

    ol, ou = jax.vmap(_atan2_elem)(
        yl.reshape(-1), yu.reshape(-1), xl.reshape(-1), xu.reshape(-1)
    )
    return Interval(ol.reshape(out_shape), ou.reshape(out_shape))


inclusion_registry[lax.atan2_p] = _inclusion_atan2_p
