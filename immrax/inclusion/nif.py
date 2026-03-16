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
                    if i == 10813 or i == 7636:
                        print("debug")
                    ans = inclusion_registry[eqn.primitive](
                        *subfuns, *invars, **bind_params
                    )
                    if eqn.primitive.multiple_results:
                        bad = any(bool(_has_nan(a)) for a in ans)
                        bad |= any(bool(_has_inf(a)) for a in ans)
                    else:
                        bad = bool(_has_nan(ans))
                        bad |= bool(_has_inf(ans))

                    if bad:
                        print("\n[NAN/inf DETECTED]")
                        print("primitive:", eqn.primitive, "id:", i)
                        print("invars:")
                        for k, v in enumerate(invars):
                            print(f"  arg{k}:", v)
                        print("out:", ans)
                        # raise FloatingPointError(f"NaN/inf detected after primitive {eqn.primitive} (id: {i})")
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


# We would like to passthrough array operations like reshaping, slicing, etc.
_add_passthrough_to_registry(lax.copy_p)
_add_passthrough_to_registry(lax.reshape_p)
_add_passthrough_to_registry(lax.slice_p)
_add_passthrough_to_registry(lax.split_p)
_add_passthrough_to_registry(lax.dynamic_slice_p)
_add_passthrough_to_registry(lax.squeeze_p)
_add_passthrough_to_registry(lax.transpose_p)
_add_passthrough_to_registry(lax.broadcast_in_dim_p)
_add_passthrough_to_registry(lax.concatenate_p)
_add_passthrough_to_registry(lax.gather_p)
_add_passthrough_to_registry(lax.scatter_p)
_add_passthrough_to_registry(lax.scatter_add_p)
_add_passthrough_to_registry(lax.scatter_max_p)
_add_passthrough_to_registry(lax.scatter_min_p)
if hasattr(lax, "select_p"):
    _add_passthrough_to_registry(lax.select_p)
if hasattr(lax, "select_n_p"):
    _add_passthrough_to_registry(lax.select_n_p)
_add_passthrough_to_registry(lax.iota_p)
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


def _inclusion_add_p(x: Interval, y: Interval) -> Interval:
    if isinstance(x, Interval) and isinstance(y, Interval):
        return Interval(x.lower + y.lower, x.upper + y.upper)
    elif isinstance(x, Interval):
        return Interval(x.lower + y, x.upper + y)
    elif isinstance(y, Interval):
        return Interval(x + y.lower, x + y.upper)
    else:
        return x + y


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
            # val = jnp.where(i == j, jnp.sqrt(A[i, i] - s), (A[i, j] - s) / L[j, j])
            if i == j:
                val = jnp.sqrt(A[i, i] - s)
            else:
                val = (A[i, j] - s) / L[j, j]
            L = L.at[i, j].set(val)
    return L

# def _manual_cholesky(A):
#     A = 0.5 * (A + A.T)  # Ensure symmetry
#     n = A.shape[0]
#     L0 = jnp.zeros_like(A)

#     def body(j, L):
#         s = L[j:, :j] @ L[j, :j]
#         ljj = jnp.sqrt(A[j, j] - s[0])

#         new_col = jnp.concatenate([
#             jnp.array([ljj], dtype=A.dtype),
#             (A[j+1:, j] - s[1:]) / ljj
#         ])

#         return L.at[j:, j].set(new_col)

#     return lax.fori_loop(0, n, body, L0)

inclusion_registry[LA.cholesky_p] = natif(_manual_cholesky)

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
