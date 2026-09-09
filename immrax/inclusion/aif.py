"""Affine-bound inclusion evaluation for JAX programs.

The public :func:`affif` transform can either concretize its result to an
:class:`Interval` (the default) or retain the internal :class:`AffineBound` so
that affine evaluations can be composed without losing source correlation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Literal as TypingLiteral, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
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
from jax._src.debugging import debug_callback_p
from jax._src.lax import linalg as LA
from jax._src.util import safe_map
from jax.extend.core import Primitive

from .affine import AffineBound, constant_affine_bound, interval_to_affine_bound
from .interval import Interval, interval

affine_inclusion_registry: dict[Primitive, Callable[..., Any]] = {}


def _is_abstract(x):
    return isinstance(x, (AffineBound, Interval))


def _template(*xs) -> AffineBound:
    for x in xs:
        if isinstance(x, AffineBound):
            return x
    raise TypeError("An affine-bound operand is required.")


def _constant(value, like: AffineBound) -> AffineBound:
    return constant_affine_bound(value, like.domain_lower, like.domain_upper)


def _promote(value, like: AffineBound) -> AffineBound:
    if isinstance(value, AffineBound):
        return value
    if isinstance(value, Interval):
        return _constant_interval_bounds(value.lower, value.upper, like)
    return _constant(value, like)


def _new(lc, lb, uc, ub, like: AffineBound) -> AffineBound:
    return AffineBound(lc, lb, uc, ub, like.domain_lower, like.domain_upper)


def _constant_interval_bounds(lower, upper, like: AffineBound) -> AffineBound:
    lower, upper = jnp.broadcast_arrays(lower, upper)
    n = like.input_size
    dtype = jnp.result_type(lower, upper, float)
    zeros = jnp.zeros(lower.shape + (n,), dtype=dtype)
    return _new(zeros, lower, zeros, upper, like)


def _select_affine(mask, yes: AffineBound, no: AffineBound) -> AffineBound:
    mask = jnp.asarray(mask)
    cmask = mask[..., None]
    return _new(
        jnp.where(cmask, yes.lower_coeff, no.lower_coeff),
        jnp.where(mask, yes.lower_bias, no.lower_bias),
        jnp.where(cmask, yes.upper_coeff, no.upper_coeff),
        jnp.where(mask, yes.upper_bias, no.upper_bias),
        yes,
    )


def _evaluate(coeff, bias, point):
    return jnp.sum(coeff * point, axis=-1) + bias


def _affine_lower_plane(terms, constant, like):
    """Back-substitute a linear expression as a lower bound."""
    out_shape = jnp.broadcast_shapes(
        jnp.shape(constant),
        *(jnp.broadcast_shapes(jnp.shape(c), z.shape) for c, z in terms),
    )
    coeff = jnp.zeros(
        out_shape + (like.input_size,), dtype=jnp.result_type(constant, float)
    )
    bias = jnp.broadcast_to(jnp.asarray(constant), out_shape)
    for c, z in terms:
        c = jnp.broadcast_to(jnp.asarray(c), out_shape)
        zl = jnp.broadcast_to(z.lower_coeff, out_shape + (like.input_size,))
        zu = jnp.broadcast_to(z.upper_coeff, out_shape + (like.input_size,))
        bl = jnp.broadcast_to(z.lower_bias, out_shape)
        bu = jnp.broadcast_to(z.upper_bias, out_shape)
        coeff = coeff + jnp.where(
            (c >= 0)[..., None], c[..., None] * zl, c[..., None] * zu
        )
        bias = bias + jnp.where(c >= 0, c * bl, c * bu)
    return coeff, bias


def _affine_upper_plane(terms, constant, like):
    """Back-substitute a linear expression as an upper bound."""
    out_shape = jnp.broadcast_shapes(
        jnp.shape(constant),
        *(jnp.broadcast_shapes(jnp.shape(c), z.shape) for c, z in terms),
    )
    coeff = jnp.zeros(
        out_shape + (like.input_size,), dtype=jnp.result_type(constant, float)
    )
    bias = jnp.broadcast_to(jnp.asarray(constant), out_shape)
    for c, z in terms:
        c = jnp.broadcast_to(jnp.asarray(c), out_shape)
        zl = jnp.broadcast_to(z.lower_coeff, out_shape + (like.input_size,))
        zu = jnp.broadcast_to(z.upper_coeff, out_shape + (like.input_size,))
        bl = jnp.broadcast_to(z.lower_bias, out_shape)
        bu = jnp.broadcast_to(z.upper_bias, out_shape)
        coeff = coeff + jnp.where(
            (c >= 0)[..., None], c[..., None] * zu, c[..., None] * zl
        )
        bias = bias + jnp.where(c >= 0, c * bu, c * bl)
    return coeff, bias


def _relax(x: AffineBound, ml, cl, mu, cu) -> AffineBound:
    """Back-substitute scalar affine lower/upper relaxations."""
    out_shape = jnp.broadcast_shapes(x.shape, jnp.shape(ml), jnp.shape(mu))
    ml, cl, mu, cu = [
        jnp.broadcast_to(jnp.asarray(v), out_shape) for v in (ml, cl, mu, cu)
    ]
    al = jnp.broadcast_to(x.lower_coeff, out_shape + (x.input_size,))
    au = jnp.broadcast_to(x.upper_coeff, out_shape + (x.input_size,))
    bl = jnp.broadcast_to(x.lower_bias, out_shape)
    bu = jnp.broadcast_to(x.upper_bias, out_shape)
    lc = jnp.where((ml >= 0)[..., None], ml[..., None] * al, ml[..., None] * au)
    lb = jnp.where(ml >= 0, ml * bl + cl, ml * bu + cl)
    uc = jnp.where((mu >= 0)[..., None], mu[..., None] * au, mu[..., None] * al)
    ub = jnp.where(mu >= 0, mu * bu + cu, mu * bl + cu)
    return _new(lc, lb, uc, ub, x)


def _degenerate_exact(result: AffineBound, x: AffineBound, f) -> AffineBound:
    degenerate = x.lower == x.upper
    exact = _constant(f(x.lower), x)
    return _select_affine(degenerate, exact, result)


def _jointly_degenerate_exact(
    result: AffineBound, operands, function
) -> AffineBound:
    """Evaluate an operation exactly where all affine operands are points."""
    values = []
    degenerate = True
    for operand in operands:
        if isinstance(operand, AffineBound):
            degenerate = jnp.logical_and(
                degenerate, operand.lower == operand.upper
            )
            values.append(operand.lower)
        else:
            values.append(operand)
    exact = _constant(function(*values), result)
    return _select_affine(degenerate, exact, result)


def _sanitize_affine(value):
    """Turn undefined numerical relaxation artifacts into sound top bounds."""
    if not isinstance(value, AffineBound):
        return value
    lower, upper = value.lower, value.upper
    invalid = jnp.logical_or(
        jnp.logical_or(jnp.isnan(lower), jnp.isnan(upper)), lower > upper
    )
    return _select_affine(
        invalid, _constant_interval_bounds(-jnp.inf, jnp.inf, value), value
    )


@overload
def affif(
    f: Callable[..., jax.Array],
    *,
    fixed_argnums: int | Sequence[int] | None = None,
    return_type: TypingLiteral["interval"] = "interval",
) -> Callable[..., Interval]: ...


@overload
def affif(
    f: Callable[..., jax.Array],
    *,
    fixed_argnums: int | Sequence[int] | None = None,
    return_type: TypingLiteral["affine"],
) -> Callable[..., AffineBound]: ...


def affif(
    f: Callable[..., jax.Array],
    *,
    fixed_argnums: int | Sequence[int] | None = None,
    return_type: TypingLiteral["interval", "affine"] = "interval",
) -> Callable[..., Interval | AffineBound]:
    """Create an affine-bound inclusion function for ``f``.

    Every non-fixed positional :class:`Interval` is lifted into one shared
    affine source domain.  Existing :class:`AffineBound` arguments are instead
    propagated in their original domain, which permits sequential affine
    composition without relifting or concretization.  Multiple affine inputs
    must share compatible source bounds, and mixing them with new ``Interval``
    inputs is currently rejected.

    Parameters
    ----------
    f:
        JAX-traceable function to transform.
    fixed_argnums:
        Positional arguments that should remain point-valued.
    return_type:
        ``"interval"`` (the backward-compatible default) concretizes affine
        outputs.  ``"affine"`` returns :class:`AffineBound` outputs and
        promotes point-valued results into the active affine source domain.

    Calls with no interval or affine positional inputs evaluate ``f`` normally,
    regardless of ``return_type``.
    """
    if return_type not in ("interval", "affine"):
        raise ValueError(
            "Unsupported affif return_type "
            f"{return_type!r}; expected 'interval' or 'affine'."
        )
    if fixed_argnums is None:
        fixed = set()
    elif isinstance(fixed_argnums, int):
        fixed = {fixed_argnums}
    else:
        fixed = set(fixed_argnums)

    def abstract_inputs(args):
        leaves = []
        for i, arg in enumerate(args):
            if i not in fixed:
                leaves.extend(jax.tree_util.tree_leaves(arg, is_leaf=_is_abstract))
        return [z for z in leaves if _is_abstract(z)]

    def validate_inputs(args):
        abstract = abstract_inputs(args)
        intervals = [z for z in abstract if isinstance(z, Interval)]
        affines = [z for z in abstract if isinstance(z, AffineBound)]
        if intervals and affines:
            raise ValueError(
                "affif cannot currently combine new Interval inputs with existing "
                "AffineBound inputs; lift all inputs into one shared affine domain."
            )
        if len(affines) < 2:
            return
        reference = affines[0]
        for candidate in affines[1:]:
            if (
                candidate.domain_lower.shape != reference.domain_lower.shape
                or candidate.domain_upper.shape != reference.domain_upper.shape
            ):
                raise ValueError(
                    "AffineBound inputs have incompatible source-domain shapes."
                )
            # At an ordinary (non-traced) call the source boxes can be checked
            # exactly.  Under an enclosing JAX transform only their static
            # shapes are available, so defer the value check.
            try:
                compatible = np.array_equal(
                    np.asarray(candidate.domain_lower),
                    np.asarray(reference.domain_lower),
                ) and np.array_equal(
                    np.asarray(candidate.domain_upper),
                    np.asarray(reference.domain_upper),
                )
            except (TypeError, jax.errors.TracerArrayConversionError):
                compatible = True
            if not compatible:
                raise ValueError(
                    "AffineBound inputs have incompatible source domains; "
                    "unrelated affine source spaces cannot be combined."
                )

    @jax.jit
    def compiled(*args, **kwargs):
        abstract = abstract_inputs(args)
        input_intervals: list[Interval] = []
        input_affines: list[AffineBound] = []
        for value in abstract:
            if isinstance(value, Interval):
                input_intervals.append(value)
            elif isinstance(value, AffineBound):
                input_affines.append(value)
        if not input_intervals and not input_affines:
            return f(*args, **kwargs)

        if input_affines:
            domain_lower = input_affines[0].domain_lower
            domain_upper = input_affines[0].domain_upper
        else:
            domain_lower = jnp.concatenate(
                [z.lower.reshape(-1) for z in input_intervals]
            )
            domain_upper = jnp.concatenate(
                [z.upper.reshape(-1) for z in input_intervals]
            )
        offset = 0

        def lift(z):
            nonlocal offset
            if isinstance(z, Interval):
                out = interval_to_affine_bound(
                    z, domain_lower, domain_upper, offset=offset
                )
                offset += z.size
                return out
            return z

        affine_args = tuple(
            arg
            if i in fixed
            else jax.tree_util.tree_map(
                lift, arg, is_leaf=lambda z: isinstance(z, Interval)
            )
            for i, arg in enumerate(args)
        )
        trace_args = jax.tree_util.tree_map(
            lambda z: z.lower if _is_abstract(z) else jnp.asarray(z),
            args,
            is_leaf=_is_abstract,
        )
        trace_kwargs = jax.tree_util.tree_map(
            lambda z: z.lower if _is_abstract(z) else z,
            kwargs,
            is_leaf=_is_abstract,
        )
        # Close over keyword arguments.  This matches natif's documented
        # convention that only positional arguments are abstract inputs and
        # avoids introducing keyword-only Jaxpr invars with no affine value.
        traced_function = lambda *positional: f(*positional, **trace_kwargs)
        closed = eqx.filter_make_jaxpr(traced_function)(*trace_args)[0]
        flat_affine_args = jax.tree_util.tree_leaves(affine_args, is_leaf=_is_abstract)
        outputs = aif_jaxpr(closed.jaxpr, closed.literals, *flat_affine_args)

        if return_type == "interval":
            def convert_output(z):
                if isinstance(z, AffineBound):
                    return z.concretize()
                if isinstance(z, Interval):
                    return z
                return interval(z)
        else:
            like = input_affines[0] if input_affines else affine_args[0]
            if not isinstance(like, AffineBound):
                like = next(z for z in flat_affine_args if isinstance(z, AffineBound))

            def convert_output(z):
                if isinstance(z, AffineBound):
                    return z
                if isinstance(z, Interval):
                    return _constant_interval_bounds(z.lower, z.upper, like)
                return _constant(z, like)

        outputs = [convert_output(z) for z in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    @wraps(f)
    def wrapped(*args, **kwargs):
        validate_inputs(args)
        return compiled(*args, **kwargs)

    return wrapped


def aif_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
    """Evaluate a Jaxpr over affine and three-valued Boolean abstractions."""
    env: dict[Var, Any] = {}

    def read(v: Atom):
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, value):
        if (
            config.enable_checks.value
            and not config.dynamic_shapes.value
            and not _is_abstract(value)
        ):
            assert typecheck(v.aval, value), (v.aval, value)
        env[v] = value

    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any(_is_abstract(v) for v in invars):
                try:
                    ans = affine_inclusion_registry[eqn.primitive](
                        *subfuns, *invars, **bind_params
                    )
                except KeyError as exc:
                    raise NotImplementedError(
                        f"{eqn.primitive} not in affine_inclusion_registry"
                    ) from exc
            else:
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
        if eqn.primitive.multiple_results:
            ans = [_sanitize_affine(value) for value in ans]
        else:
            ans = _sanitize_affine(ans)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)
    return safe_map(read, jaxpr.outvars)


# ---------------------------------------------------------------------------
# Exact affine arithmetic


def _interval_add(x, y):
    x = x if isinstance(x, Interval) else interval(x)
    y = y if isinstance(y, Interval) else interval(y)
    return Interval(x.lower + y.lower, x.upper + y.upper)


def _interval_neg(x):
    x = x if isinstance(x, Interval) else interval(x)
    return Interval(-x.upper, -x.lower)


def _interval_mul(x, y):
    x = x if isinstance(x, Interval) else interval(x)
    y = y if isinstance(y, Interval) else interval(y)
    products = jnp.stack(
        (
            x.lower * y.lower,
            x.lower * y.upper,
            x.upper * y.lower,
            x.upper * y.upper,
        )
    )
    return Interval(jnp.min(products, axis=0), jnp.max(products, axis=0))


def _interval_reciprocal(x):
    x = x if isinstance(x, Interval) else interval(x)
    crosses_zero = jnp.logical_and(x.lower <= 0, x.upper >= 0)
    safe_lower = jnp.where(x.lower == 0, 1, x.lower)
    safe_upper = jnp.where(x.upper == 0, 1, x.upper)
    lower = jnp.minimum(1 / safe_lower, 1 / safe_upper)
    upper = jnp.maximum(1 / safe_lower, 1 / safe_upper)
    return Interval(
        jnp.where(crosses_zero, -jnp.inf, lower),
        jnp.where(crosses_zero, jnp.inf, upper),
    )


def _add(x, y):
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return _interval_add(x, y)
        return x + y
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    n = like.input_size
    return _new(
        jnp.broadcast_to(x.lower_coeff, shape + (n,))
        + jnp.broadcast_to(y.lower_coeff, shape + (n,)),
        jnp.broadcast_to(x.lower_bias, shape) + jnp.broadcast_to(y.lower_bias, shape),
        jnp.broadcast_to(x.upper_coeff, shape + (n,))
        + jnp.broadcast_to(y.upper_coeff, shape + (n,)),
        jnp.broadcast_to(x.upper_bias, shape) + jnp.broadcast_to(y.upper_bias, shape),
        like,
    )


def _neg(x):
    if isinstance(x, Interval):
        return _interval_neg(x)
    if not isinstance(x, AffineBound):
        return lax.neg(x)
    return _new(-x.upper_coeff, -x.upper_bias, -x.lower_coeff, -x.lower_bias, x)


def _sub(x, y):
    return _add(x, _neg(y))


def _scale(x: AffineBound, c) -> AffineBound:
    shape = jnp.broadcast_shapes(x.shape, jnp.shape(c))
    c = jnp.broadcast_to(jnp.asarray(c), shape)
    n = x.input_size
    al = jnp.broadcast_to(x.lower_coeff, shape + (n,))
    au = jnp.broadcast_to(x.upper_coeff, shape + (n,))
    bl = jnp.broadcast_to(x.lower_bias, shape)
    bu = jnp.broadcast_to(x.upper_bias, shape)
    return _new(
        jnp.where((c >= 0)[..., None], c[..., None] * al, c[..., None] * au),
        jnp.where(c >= 0, c * bl, c * bu),
        jnp.where((c >= 0)[..., None], c[..., None] * au, c[..., None] * al),
        jnp.where(c >= 0, c * bu, c * bl),
        x,
    )


def _mul(x, y):
    if (
        isinstance(x, AffineBound)
        and not isinstance(y, (AffineBound, Interval))
    ):
        return _scale(x, y)
    if (
        isinstance(y, AffineBound)
        and not isinstance(x, (AffineBound, Interval))
    ):
        return _scale(y, x)
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return _interval_mul(x, y)
        return x * y

    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    pl, pu, ql, qu = jnp.broadcast_arrays(x.lower, x.upper, y.lower, y.upper)
    # McCormick candidates, back-substituted through existing affine forms.
    l1c, l1b = _affine_lower_plane(((ql, x), (pl, y)), -pl * ql, like)
    l2c, l2b = _affine_lower_plane(((qu, x), (pu, y)), -pu * qu, like)
    u1c, u1b = _affine_upper_plane(((ql, x), (pu, y)), -pu * ql, like)
    u2c, u2b = _affine_upper_plane(((qu, x), (pl, y)), -pl * qu, like)
    center = (like.domain_lower + like.domain_upper) / 2
    choose_l1 = _evaluate(l1c, l1b, center) >= _evaluate(l2c, l2b, center)
    choose_u1 = _evaluate(u1c, u1b, center) <= _evaluate(u2c, u2b, center)
    return _new(
        jnp.where(choose_l1[..., None], l1c, l2c),
        jnp.where(choose_l1, l1b, l2b),
        jnp.where(choose_u1[..., None], u1c, u2c),
        jnp.where(choose_u1, u1b, u2b),
        like,
    )


def _reciprocal(x):
    if isinstance(x, Interval):
        return _interval_reciprocal(x)
    if not isinstance(x, AffineBound):
        return 1 / x
    l, u = x.lower, x.upper
    safe_l = jnp.where(l == 0, 1, l)
    safe_u = jnp.where(u == 0, 1, u)
    ms = -1 / (safe_l * safe_u)
    t = jnp.where(l > 0, jnp.sqrt(safe_l * safe_u), -jnp.sqrt(safe_l * safe_u))
    sec_c = 1 / safe_l - ms * safe_l
    tan_c = 1 / t - (-1 / t**2) * t
    mt = -1 / t**2
    positive = l > 0
    lower = _relax(
        x,
        jnp.where(positive, mt, ms),
        jnp.where(positive, tan_c, sec_c),
        jnp.where(positive, ms, mt),
        jnp.where(positive, sec_c, tan_c),
    )
    lower = _degenerate_exact(lower, x, lambda value: 1 / value)
    valid = jnp.logical_or(l > 0, u < 0)
    top = _constant_interval_bounds(-jnp.inf, jnp.inf, x)
    return _select_affine(valid, lower, top)


def _div(x, y):
    if isinstance(y, (AffineBound, Interval)) and isinstance(
        x, (AffineBound, Interval)
    ):
        if isinstance(x, Interval) and isinstance(y, Interval):
            return _interval_mul(x, _interval_reciprocal(y))
        like = _template(x, y)
        return _mul(_promote(x, like), _reciprocal(_promote(y, like)))
    if isinstance(y, AffineBound):
        return _mul(x, _reciprocal(y))
    if isinstance(x, AffineBound):
        return _scale(x, 1 / y)
    if isinstance(x, Interval) or isinstance(y, Interval):
        return _interval_mul(x, _interval_reciprocal(y))
    return x / y


affine_inclusion_registry[lax.add_p] = _add
affine_inclusion_registry[ad_util.add_any_p] = _add
affine_inclusion_registry[lax.sub_p] = _sub
affine_inclusion_registry[lax.neg_p] = _neg
affine_inclusion_registry[lax.mul_p] = _mul
affine_inclusion_registry[lax.div_p] = _div

AffineBound.__add__ = _add
AffineBound.__radd__ = lambda self, other: _add(other, self)
AffineBound.__sub__ = _sub
AffineBound.__rsub__ = lambda self, other: _sub(other, self)
AffineBound.__mul__ = _mul
AffineBound.__rmul__ = lambda self, other: _mul(other, self)
AffineBound.__truediv__ = _div
AffineBound.__rtruediv__ = lambda self, other: _div(other, self)
AffineBound.__neg__ = _neg


# ---------------------------------------------------------------------------
# Scalar nonlinear relaxations


def _exp(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.exp(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    d = u - l
    ms = jnp.where(
        d == 0, jnp.exp(l), (jnp.exp(u) - jnp.exp(l)) / jnp.where(d == 0, 1, d)
    )
    t = jnp.log(ms)
    out = _relax(x, ms, jnp.exp(t) - ms * t, ms, jnp.exp(l) - ms * l)
    return _degenerate_exact(out, x, jnp.exp)


def _log(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.log(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    sl = jnp.where(l > 0, l, 1)
    su = jnp.where(u > 0, u, 1)
    d = su - sl
    ms = jnp.where(
        d == 0, 1 / sl, (jnp.log(su) - jnp.log(sl)) / jnp.where(d == 0, 1, d)
    )
    t = 1 / ms
    out = _relax(x, ms, jnp.log(sl) - ms * sl, ms, jnp.log(t) - ms * t)
    out = _degenerate_exact(out, x, jnp.log)
    top = _constant_interval_bounds(-jnp.inf, jnp.inf, x)
    return _select_affine(l > 0, out, top)


def _log1p(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.log1p(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    sl = jnp.where(l > -1, l, 0)
    su = jnp.where(u > -1, u, 0)
    d = su - sl
    ms = jnp.where(
        d == 0, 1 / (1 + sl), (jnp.log1p(su) - jnp.log1p(sl)) / jnp.where(d == 0, 1, d)
    )
    t = 1 / ms - 1
    out = _relax(x, ms, jnp.log1p(sl) - ms * sl, ms, jnp.log1p(t) - ms * t)
    out = _degenerate_exact(out, x, jnp.log1p)
    return _select_affine(l > -1, out, _constant_interval_bounds(-jnp.inf, jnp.inf, x))


def _sqrt(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.sqrt(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    sl, su = jnp.maximum(l, 0), jnp.maximum(u, 0)
    d = su - sl
    ms = jnp.where(d == 0, 1, (jnp.sqrt(su) - jnp.sqrt(sl)) / jnp.where(d == 0, 1, d))
    t = 1 / (4 * ms**2)
    out = _relax(x, ms, jnp.sqrt(sl) - ms * sl, ms, jnp.sqrt(t) - ms * t)
    out = _degenerate_exact(out, x, jnp.sqrt)
    return _select_affine(l >= 0, out, _constant_interval_bounds(-jnp.inf, jnp.inf, x))


def _square(x):
    if not isinstance(x, AffineBound):
        return lax.square(x)
    l, u = x.lower, x.upper
    ms = l + u
    sec_c = -l * u
    tangent_c = -((l + u) ** 2) / 4
    crossing = jnp.logical_and(l < 0, u > 0)
    out = _relax(
        x, jnp.where(crossing, 0, ms), jnp.where(crossing, 0, tangent_c), ms, sec_c
    )
    return _degenerate_exact(out, x, jnp.square)


def _integer_pow(x, exponent: int | None = None, *, y: int | None = None):
    if exponent is None:
        exponent = y
    if not isinstance(x, AffineBound):
        return lax.integer_pow(x, exponent)
    if exponent == 0:
        return _constant(jnp.ones(x.shape, dtype=x.dtype), x)
    if exponent == 1:
        return x
    if exponent == 2:
        return _square(x)
    if exponent < 0:
        result = _integer_pow(_reciprocal(x), -exponent)
        valid = jnp.logical_or(x.lower > 0, x.upper < 0)
        return _select_affine(
            valid, result, _constant_interval_bounds(-jnp.inf, jnp.inf, x)
        )

    l, u = x.lower, x.upper
    d = u - l
    lp, up = l**exponent, u**exponent
    ms = jnp.where(
        d == 0, exponent * l ** (exponent - 1), (up - lp) / jnp.where(d == 0, 1, d)
    )
    sign = jnp.where(u <= 0, -1.0, 1.0)
    t = sign * jnp.abs(ms / exponent) ** (1 / (exponent - 1))
    tangent_c = t**exponent - ms * t
    secant_c = lp - ms * l
    even = exponent % 2 == 0
    crossing = jnp.logical_and(l < 0, u > 0)
    convex = jnp.logical_or(l >= 0, even)
    regular = _relax(
        x,
        jnp.where(convex, ms, ms),
        jnp.where(convex, tangent_c, secant_c),
        jnp.where(convex, ms, ms),
        jnp.where(convex, secant_c, tangent_c),
    )
    if even:
        cross = _relax(x, 0, 0, ms, secant_c)
    else:
        cross = _constant_interval_bounds(lp, up, x)
    return _degenerate_exact(
        _select_affine(crossing, cross, regular), x, lambda z: z**exponent
    )


def _abs(x):
    if not isinstance(x, AffineBound):
        return lax.abs(x)
    l, u = x.lower, x.upper
    crossing = jnp.logical_and(l < 0, u > 0)
    ms = (u + l) / jnp.where(u == l, 1, u - l)
    sec_c = -2 * l * u / jnp.where(u == l, 1, u - l)
    positive = x
    negative = _neg(x)
    cross = _relax(x, 0, 0, ms, sec_c)
    out = _select_affine(crossing, cross, _select_affine(l >= 0, positive, negative))
    return _degenerate_exact(out, x, jnp.abs)


affine_inclusion_registry[lax.exp_p] = _exp
affine_inclusion_registry[lax.log_p] = _log
affine_inclusion_registry[lax.log1p_p] = _log1p
affine_inclusion_registry[lax.sqrt_p] = _sqrt
affine_inclusion_registry[lax.square_p] = _square
affine_inclusion_registry[lax.integer_pow_p] = _integer_pow
affine_inclusion_registry[lax.abs_p] = _abs
AffineBound.__pow__ = _integer_pow
AffineBound.__abs__ = _abs


def _tanh(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.tanh(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    d = u - l
    ms = jnp.where(
        d == 0,
        1 - jnp.tanh(l) ** 2,
        (jnp.tanh(u) - jnp.tanh(l)) / jnp.where(d == 0, 1, d),
    )
    root = jnp.sqrt(jnp.maximum(0, 1 - ms))
    sign = jnp.where(u <= 0, -1.0, 1.0)
    t = sign * jnp.arctanh(root)
    sec_c = jnp.tanh(l) - ms * l
    tan_c = jnp.tanh(t) - ms * t
    negative = u <= 0
    regular = _relax(
        x,
        ms,
        jnp.where(negative, tan_c, sec_c),
        ms,
        jnp.where(negative, sec_c, tan_c),
    )
    crossing = jnp.logical_and(l < 0, u > 0)
    fallback = _constant_interval_bounds(jnp.tanh(l), jnp.tanh(u), x)
    return _degenerate_exact(_select_affine(crossing, fallback, regular), x, jnp.tanh)


def _periodic_extremum_present(l, u, origin, period):
    return jnp.ceil((l - origin) / period) <= jnp.floor((u - origin) / period)


def _sin(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.sin(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    d = u - l
    k = jnp.floor(l / jnp.pi)
    cell_end = (k + 1) * jnp.pi
    same_cell = u <= cell_end
    ms = jnp.where(
        d == 0, jnp.cos(l), (jnp.sin(u) - jnp.sin(l)) / jnp.where(d == 0, 1, d)
    )
    acos_ms = jnp.arccos(jnp.clip(ms, -1, 1))
    even = jnp.mod(k, 2) == 0
    t = jnp.where(even, k * jnp.pi + acos_ms, (k + 1) * jnp.pi - acos_ms)
    sec_c = jnp.sin(l) - ms * l
    tan_c = jnp.sin(t) - ms * t
    regular = _relax(
        x,
        ms,
        jnp.where(even, sec_c, tan_c),
        ms,
        jnp.where(even, tan_c, sec_c),
    )
    lower = jnp.where(
        _periodic_extremum_present(l, u, -jnp.pi / 2, 2 * jnp.pi),
        -1.0,
        jnp.minimum(jnp.sin(l), jnp.sin(u)),
    )
    upper = jnp.where(
        _periodic_extremum_present(l, u, jnp.pi / 2, 2 * jnp.pi),
        1.0,
        jnp.maximum(jnp.sin(l), jnp.sin(u)),
    )
    fallback = _constant_interval_bounds(lower, upper, x)
    return _degenerate_exact(_select_affine(same_cell, regular, fallback), x, jnp.sin)


def _cos(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.cos(x, accuracy=accuracy)
    return _sin(_add(x, jnp.asarray(jnp.pi / 2, dtype=x.dtype)), accuracy=accuracy)


def _tan(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.tan(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    d = u - l
    k = jnp.floor((l + jnp.pi / 2) / jnp.pi)
    center = k * jnp.pi
    valid = jnp.logical_and(l > center - jnp.pi / 2, u < center + jnp.pi / 2)
    ms = jnp.where(
        d == 0, 1 / jnp.cos(l) ** 2, (jnp.tan(u) - jnp.tan(l)) / jnp.where(d == 0, 1, d)
    )
    delta = jnp.arctan(jnp.sqrt(jnp.maximum(0, ms - 1)))
    left = u <= center
    right = l >= center
    t = jnp.where(left, center - delta, center + delta)
    sec_c = jnp.tan(l) - ms * l
    tan_c = jnp.tan(t) - ms * t
    regular = _relax(
        x,
        ms,
        jnp.where(left, sec_c, tan_c),
        ms,
        jnp.where(left, tan_c, sec_c),
    )
    mixed = _constant_interval_bounds(jnp.tan(l), jnp.tan(u), x)
    in_branch = _select_affine(jnp.logical_or(left, right), regular, mixed)
    in_branch = _degenerate_exact(in_branch, x, jnp.tan)
    out = _select_affine(
        valid, in_branch, _constant_interval_bounds(-jnp.inf, jnp.inf, x)
    )
    return out


def _atan(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.atan(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    d = u - l
    ms = jnp.where(
        d == 0,
        1 / (1 + l**2),
        (jnp.arctan(u) - jnp.arctan(l)) / jnp.where(d == 0, 1, d),
    )
    sign = jnp.where(u <= 0, -1.0, 1.0)
    t = sign * jnp.sqrt(jnp.maximum(0, 1 / ms - 1))
    sec_c = jnp.arctan(l) - ms * l
    tan_c = jnp.arctan(t) - ms * t
    negative = u <= 0
    regular = _relax(
        x,
        ms,
        jnp.where(negative, tan_c, sec_c),
        ms,
        jnp.where(negative, sec_c, tan_c),
    )
    crossing = jnp.logical_and(l < 0, u > 0)
    return _degenerate_exact(
        _select_affine(
            crossing,
            _constant_interval_bounds(jnp.arctan(l), jnp.arctan(u), x),
            regular,
        ),
        x,
        jnp.arctan,
    )


def _asin(x, accuracy=None):
    if not isinstance(x, AffineBound):
        return lax.asin(x, accuracy=accuracy)
    l, u = x.lower, x.upper
    sl, su = jnp.clip(l, -1, 1), jnp.clip(u, -1, 1)
    d = su - sl
    ms = jnp.where(
        d == 0,
        1 / jnp.sqrt(jnp.maximum(1e-30, 1 - sl**2)),
        (jnp.arcsin(su) - jnp.arcsin(sl)) / jnp.where(d == 0, 1, d),
    )
    sign = jnp.where(u <= 0, -1.0, 1.0)
    t = sign * jnp.sqrt(jnp.maximum(0, 1 - 1 / ms**2))
    sec_c = jnp.arcsin(sl) - ms * sl
    tan_c = jnp.arcsin(t) - ms * t
    negative = u <= 0
    regular = _relax(
        x,
        ms,
        jnp.where(negative, sec_c, tan_c),
        ms,
        jnp.where(negative, tan_c, sec_c),
    )
    crossing = jnp.logical_and(l < 0, u > 0)
    in_domain = _select_affine(
        crossing,
        _constant_interval_bounds(jnp.arcsin(sl), jnp.arcsin(su), x),
        regular,
    )
    in_domain = _degenerate_exact(in_domain, x, jnp.arcsin)
    out = _select_affine(
        jnp.logical_and(l >= -1, u <= 1),
        in_domain,
        _constant_interval_bounds(-jnp.inf, jnp.inf, x),
    )
    return out


def _pow(x, y):
    if isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        try:
            y_array = np.asarray(y)
            if (
                y_array.ndim == 0
                and np.isfinite(y_array)
                and float(y_array).is_integer()
            ):
                return _integer_pow(x, int(y_array))
        except jax.errors.TracerArrayConversionError:
            y_array = None
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        return lax.pow(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    positive_result = _exp(_mul(y, _log(x)))
    xl, xu, yl, yu = jnp.broadcast_arrays(x.lower, x.upper, y.lower, y.upper)
    strictly_positive = xl > 0
    zero_base = jnp.logical_and(xl == 0, xu > 0)
    positive_exp = yl > 0
    max_zero = jnp.where(xu >= 1, xu**yu, xu**yl)
    zero_result = _constant_interval_bounds(jnp.zeros_like(max_zero), max_zero, like)
    fallback = _constant_interval_bounds(-jnp.inf, jnp.inf, like)
    result = _select_affine(
        strictly_positive,
        positive_result,
        _select_affine(jnp.logical_and(zero_base, positive_exp), zero_result, fallback),
    )
    return _jointly_degenerate_exact(result, (x, y), jnp.power)


affine_inclusion_registry[lax.tanh_p] = _tanh
affine_inclusion_registry[lax.sin_p] = _sin
affine_inclusion_registry[lax.cos_p] = _cos
affine_inclusion_registry[lax.tan_p] = _tan
affine_inclusion_registry[lax.atan_p] = _atan
affine_inclusion_registry[lax.asin_p] = _asin
affine_inclusion_registry[lax.pow_p] = _pow


# ---------------------------------------------------------------------------
# Min/max, comparisons, Boolean abstraction, and selection


def _maximum(x, y):
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        return lax.max(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    general = _scale(_add(_add(x, y), _abs(_sub(x, y))), 0.5)
    return _select_affine(
        x.lower >= y.upper, x, _select_affine(y.lower >= x.upper, y, general)
    )


def _minimum(x, y):
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        return lax.min(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    general = _scale(_sub(_add(x, y), _abs(_sub(x, y))), 0.5)
    return _select_affine(
        x.lower >= y.upper, y, _select_affine(y.lower >= x.upper, x, general)
    )


def _clamp(minimum, x, maximum):
    return _minimum(_maximum(x, minimum), maximum)


def _difference_bounds(x, y):
    if isinstance(x, AffineBound) or isinstance(y, AffineBound):
        d = _sub(x, y)
        return d.lower, d.upper
    x = interval(x) if isinstance(x, Interval) else Interval(x, x)
    y = interval(y) if isinstance(y, Interval) else Interval(y, y)
    return x.lower - y.upper, x.upper - y.lower


def _lt(x, y):
    l, u = _difference_bounds(x, y)
    return Interval(u < 0, l < 0)


def _le(x, y):
    l, u = _difference_bounds(x, y)
    return Interval(u <= 0, l <= 0)


def _gt(x, y):
    return _lt(y, x)


def _ge(x, y):
    return _le(y, x)


def _eq(x, y):
    l, u = _difference_bounds(x, y)
    return Interval(jnp.logical_and(l == 0, u == 0), jnp.logical_and(l <= 0, u >= 0))


def _not(x):
    x = x if isinstance(x, Interval) else Interval(x, x)
    return Interval(jnp.logical_not(x.upper), jnp.logical_not(x.lower))


def _ne(x, y):
    return _not(_eq(x, y))


def _and(x, y):
    x = x if isinstance(x, Interval) else Interval(x, x)
    y = y if isinstance(y, Interval) else Interval(y, y)
    return Interval(
        jnp.logical_and(x.lower, y.lower), jnp.logical_and(x.upper, y.upper)
    )


def _or(x, y):
    x = x if isinstance(x, Interval) else Interval(x, x)
    y = y if isinstance(y, Interval) else Interval(y, y)
    return Interval(jnp.logical_or(x.lower, y.lower), jnp.logical_or(x.upper, y.upper))


def _hull(x, y, like=None):
    if isinstance(x, AffineBound) or isinstance(y, AffineBound):
        if like is None:
            like = _template(x, y)
        x, y = _promote(x, like), _promote(y, like)
        return _constant_interval_bounds(
            jnp.minimum(x.lower, y.lower), jnp.maximum(x.upper, y.upper), like
        )
    x, y = interval(x), interval(y)
    return Interval(jnp.minimum(x.lower, y.lower), jnp.maximum(x.upper, y.upper))


def _select(pred, on_true, on_false):
    pred = pred if isinstance(pred, Interval) else Interval(pred, pred)
    if isinstance(on_true, AffineBound) or isinstance(on_false, AffineBound):
        like = _template(on_true, on_false)
        on_true, on_false = _promote(on_true, like), _promote(on_false, like)
        hull = _hull(on_true, on_false, like)
        return _select_affine(
            pred.lower,
            on_true,
            _select_affine(jnp.logical_not(pred.upper), on_false, hull),
        )
    on_true, on_false = interval(on_true), interval(on_false)
    hull = _hull(on_true, on_false)
    return Interval(
        jnp.where(
            pred.lower,
            on_true.lower,
            jnp.where(jnp.logical_not(pred.upper), on_false.lower, hull.lower),
        ),
        jnp.where(
            pred.lower,
            on_true.upper,
            jnp.where(jnp.logical_not(pred.upper), on_false.upper, hull.upper),
        ),
    )


def _select_n(which, *cases):
    if isinstance(which, AffineBound):
        which = which.concretize()
    elif not isinstance(which, Interval):
        which = Interval(which, which)
    if len(cases) == 2 and which.lower.dtype == jnp.bool_:
        return _select(which, cases[1], cases[0])
    if any(isinstance(c, AffineBound) for c in cases):
        like = _template(*cases)
        cases = [_promote(c, like) for c in cases]
        result = cases[0]
        unique = which.lower == which.upper
        for i, case in enumerate(cases):
            result = _select_affine(
                jnp.logical_and(unique, which.lower == i), case, result
            )
        lower = jnp.full_like(cases[0].lower, jnp.inf)
        upper = jnp.full_like(cases[0].upper, -jnp.inf)
        for i, case in enumerate(cases):
            possible = jnp.logical_and(which.lower <= i, i <= which.upper)
            lower = jnp.where(possible, jnp.minimum(lower, case.lower), lower)
            upper = jnp.where(possible, jnp.maximum(upper, case.upper), upper)
        return _select_affine(
            unique, result, _constant_interval_bounds(lower, upper, like)
        )
    cases = [interval(c) for c in cases]
    lower = jnp.full_like(cases[0].lower, jnp.inf)
    upper = jnp.full_like(cases[0].upper, -jnp.inf)
    for i, case in enumerate(cases):
        possible = jnp.logical_and(which.lower <= i, i <= which.upper)
        lower = jnp.where(possible, jnp.minimum(lower, case.lower), lower)
        upper = jnp.where(possible, jnp.maximum(upper, case.upper), upper)
    return Interval(lower, upper)


def _convert_element_type(x, **kwargs):
    primitive = lax.convert_element_type_p
    if isinstance(x, Interval):
        return Interval(
            primitive.bind(x.lower, **kwargs), primitive.bind(x.upper, **kwargs)
        )
    if not isinstance(x, AffineBound):
        return primitive.bind(x, **kwargs)
    # Affine coefficients stay floating point; the represented values/biases
    # follow JAX's requested value-preserving floating conversion.
    new_dtype = kwargs.get("new_dtype", x.dtype)
    if not jnp.issubdtype(new_dtype, jnp.floating):
        return _constant_interval_bounds(
            primitive.bind(x.lower, **kwargs),
            primitive.bind(x.upper, **kwargs),
            x,
        )
    return _new(
        primitive.bind(x.lower_coeff, **kwargs),
        primitive.bind(x.lower_bias, **kwargs),
        primitive.bind(x.upper_coeff, **kwargs),
        primitive.bind(x.upper_bias, **kwargs),
        x,
    )


for primitive, rule in (
    (lax.max_p, _maximum),
    (lax.min_p, _minimum),
    (lax.lt_p, _lt),
    (lax.le_p, _le),
    (lax.gt_p, _gt),
    (lax.ge_p, _ge),
    (lax.eq_p, _eq),
    (lax.ne_p, _ne),
    (lax.and_p, _and),
    (lax.or_p, _or),
    (lax.not_p, _not),
    (lax.convert_element_type_p, _convert_element_type),
):
    affine_inclusion_registry[primitive] = rule
if hasattr(lax, "lt_to_p"):
    affine_inclusion_registry[lax.lt_to_p] = _lt
if hasattr(lax, "select_p"):
    affine_inclusion_registry[lax.select_p] = _select
if hasattr(lax, "select_n_p"):
    affine_inclusion_registry[lax.select_n_p] = _select_n


# ---------------------------------------------------------------------------
# Tensor transformations and reductions


def _unary_linear_primitive(primitive, x, **kwargs):
    if isinstance(x, Interval):
        return Interval(
            primitive.bind(x.lower, **kwargs), primitive.bind(x.upper, **kwargs)
        )
    if not isinstance(x, AffineBound):
        return primitive.bind(x, **kwargs)
    apply_coeff = lambda a: jax.vmap(
        lambda ai: primitive.bind(ai, **kwargs), in_axes=-1, out_axes=-1
    )(a)
    return _new(
        apply_coeff(x.lower_coeff),
        primitive.bind(x.lower_bias, **kwargs),
        apply_coeff(x.upper_coeff),
        primitive.bind(x.upper_bias, **kwargs),
        x,
    )


def _copy(x):
    if isinstance(x, AffineBound):
        return _new(
            lax.copy(x.lower_coeff),
            lax.copy(x.lower_bias),
            lax.copy(x.upper_coeff),
            lax.copy(x.upper_bias),
            x,
        )
    if isinstance(x, Interval):
        return Interval(lax.copy(x.lower), lax.copy(x.upper))
    return lax.copy(x)


def _stop_gradient(x):
    if isinstance(x, AffineBound):
        return _new(
            lax.stop_gradient(x.lower_coeff),
            lax.stop_gradient(x.lower_bias),
            lax.stop_gradient(x.upper_coeff),
            lax.stop_gradient(x.upper_bias),
            x,
        )
    if isinstance(x, Interval):
        return Interval(lax.stop_gradient(x.lower), lax.stop_gradient(x.upper))
    return lax.stop_gradient(x)


def _reshape(x, **kwargs):
    return _unary_linear_primitive(lax.reshape_p, x, **kwargs)


def _slice(x, **kwargs):
    return _unary_linear_primitive(lax.slice_p, x, **kwargs)


def _squeeze(x, **kwargs):
    return _unary_linear_primitive(lax.squeeze_p, x, **kwargs)


def _transpose(x, **kwargs):
    return _unary_linear_primitive(lax.transpose_p, x, **kwargs)


def _broadcast_in_dim(x, **kwargs):
    return _unary_linear_primitive(lax.broadcast_in_dim_p, x, **kwargs)


def _split(x, **kwargs):
    if not isinstance(x, AffineBound):
        if isinstance(x, Interval):
            return [
                Interval(l, u)
                for l, u in zip(
                    lax.split_p.bind(x.lower, **kwargs),
                    lax.split_p.bind(x.upper, **kwargs),
                )
            ]
        return lax.split_p.bind(x, **kwargs)
    lbs = lax.split_p.bind(x.lower_bias, **kwargs)
    ubs = lax.split_p.bind(x.upper_bias, **kwargs)
    lcs = jax.vmap(lambda a: lax.split_p.bind(a, **kwargs), in_axes=-1, out_axes=-1)(
        x.lower_coeff
    )
    ucs = jax.vmap(lambda a: lax.split_p.bind(a, **kwargs), in_axes=-1, out_axes=-1)(
        x.upper_coeff
    )
    return [_new(lc, lb, uc, ub, x) for lc, lb, uc, ub in zip(lcs, lbs, ucs, ubs)]


def _concatenate(*xs, **kwargs):
    if not any(isinstance(x, AffineBound) for x in xs):
        if any(isinstance(x, Interval) for x in xs):
            xs = [interval(x) for x in xs]
            return Interval(
                lax.concatenate_p.bind(*[x.lower for x in xs], **kwargs),
                lax.concatenate_p.bind(*[x.upper for x in xs], **kwargs),
            )
        return lax.concatenate_p.bind(*xs, **kwargs)
    like = _template(*xs)
    xs = [_promote(x, like) for x in xs]
    apply_coeff = lambda arrays: jax.vmap(
        lambda *parts: lax.concatenate_p.bind(*parts, **kwargs),
        in_axes=tuple(-1 for _ in arrays),
        out_axes=-1,
    )(*arrays)
    return _new(
        apply_coeff([x.lower_coeff for x in xs]),
        lax.concatenate_p.bind(*[x.lower_bias for x in xs], **kwargs),
        apply_coeff([x.upper_coeff for x in xs]),
        lax.concatenate_p.bind(*[x.upper_bias for x in xs], **kwargs),
        like,
    )


def _dynamic_slice(x, *start_indices, **kwargs):
    if any(_is_abstract(i) for i in start_indices):
        if not isinstance(x, AffineBound):
            x = interval(x)
            shape = tuple(kwargs["slice_sizes"])
            return Interval(
                jnp.full(shape, jnp.min(x.lower)), jnp.full(shape, jnp.max(x.upper))
            )
        shape = tuple(kwargs["slice_sizes"])
        return _constant_interval_bounds(
            jnp.full(shape, jnp.min(x.lower)), jnp.full(shape, jnp.max(x.upper)), x
        )
    if isinstance(x, Interval):
        return Interval(
            lax.dynamic_slice_p.bind(x.lower, *start_indices, **kwargs),
            lax.dynamic_slice_p.bind(x.upper, *start_indices, **kwargs),
        )
    if not isinstance(x, AffineBound):
        return lax.dynamic_slice_p.bind(x, *start_indices, **kwargs)
    apply_coeff = lambda a: jax.vmap(
        lambda ai: lax.dynamic_slice_p.bind(ai, *start_indices, **kwargs),
        in_axes=-1,
        out_axes=-1,
    )(a)
    return _new(
        apply_coeff(x.lower_coeff),
        lax.dynamic_slice_p.bind(x.lower_bias, *start_indices, **kwargs),
        apply_coeff(x.upper_coeff),
        lax.dynamic_slice_p.bind(x.upper_bias, *start_indices, **kwargs),
        x,
    )


def _gather(operand, indices, **kwargs):
    if _is_abstract(indices):
        like = operand if isinstance(operand, AffineBound) else _template(indices)
        index_value = indices.lower.astype(jnp.int32)
        out_shape = lax.gather_p.bind(
            operand.lower_bias
            if isinstance(operand, AffineBound)
            else jnp.asarray(operand),
            index_value,
            **kwargs,
        ).shape
        source = (
            operand.concretize()
            if isinstance(operand, AffineBound)
            else interval(operand)
        )
        return _constant_interval_bounds(
            jnp.full(out_shape, jnp.min(source.lower)),
            jnp.full(out_shape, jnp.max(source.upper)),
            like,
        )
    if isinstance(operand, Interval):
        return Interval(
            lax.gather_p.bind(operand.lower, indices, **kwargs),
            lax.gather_p.bind(operand.upper, indices, **kwargs),
        )
    if not isinstance(operand, AffineBound):
        return lax.gather_p.bind(operand, indices, **kwargs)
    apply_coeff = lambda a: jax.vmap(
        lambda ai: lax.gather_p.bind(ai, indices, **kwargs), in_axes=-1, out_axes=-1
    )(a)
    return _new(
        apply_coeff(operand.lower_coeff),
        lax.gather_p.bind(operand.lower_bias, indices, **kwargs),
        apply_coeff(operand.upper_coeff),
        lax.gather_p.bind(operand.upper_bias, indices, **kwargs),
        operand,
    )


def _pad(x, padding_value, **kwargs):
    if not isinstance(x, AffineBound) and not isinstance(padding_value, AffineBound):
        x, padding_value = interval(x), interval(padding_value)
        return Interval(
            lax.pad_p.bind(x.lower, padding_value.lower, **kwargs),
            lax.pad_p.bind(x.upper, padding_value.upper, **kwargs),
        )
    like = _template(x, padding_value)
    x, padding_value = _promote(x, like), _promote(padding_value, like)
    apply_coeff = lambda a, p: jax.vmap(
        lambda ai, pi: lax.pad_p.bind(ai, pi, **kwargs), in_axes=(-1, -1), out_axes=-1
    )(a, p)
    return _new(
        apply_coeff(x.lower_coeff, padding_value.lower_coeff),
        lax.pad_p.bind(x.lower_bias, padding_value.lower_bias, **kwargs),
        apply_coeff(x.upper_coeff, padding_value.upper_coeff),
        lax.pad_p.bind(x.upper_bias, padding_value.upper_bias, **kwargs),
        like,
    )


def _scatter_linear(primitive, operand, indices, updates, **kwargs):
    if _is_abstract(indices):
        # Conservative uncertain-index fallback.  Replacement may choose any
        # update; scatter-add may add every update to every destination.
        like = _template(operand, updates, indices)
        operand, updates = _promote(operand, like), _promote(updates, like)
        if primitive is lax.scatter_add_p:
            dl = jnp.sum(jnp.minimum(0, updates.lower))
            du = jnp.sum(jnp.maximum(0, updates.upper))
            return _add(operand, _constant_interval_bounds(dl, du, like))
        return _constant_interval_bounds(
            jnp.minimum(operand.lower, jnp.min(updates.lower)),
            jnp.maximum(operand.upper, jnp.max(updates.upper)),
            like,
        )
    if not isinstance(operand, AffineBound) and not isinstance(updates, AffineBound):
        return primitive.bind(operand, indices, updates, **kwargs)
    like = _template(operand, updates)
    operand, updates = _promote(operand, like), _promote(updates, like)
    apply_coeff = lambda a, u: jax.vmap(
        lambda ai, ui: primitive.bind(ai, indices, ui, **kwargs),
        in_axes=(-1, -1),
        out_axes=-1,
    )(a, u)
    return _new(
        apply_coeff(operand.lower_coeff, updates.lower_coeff),
        primitive.bind(operand.lower_bias, indices, updates.lower_bias, **kwargs),
        apply_coeff(operand.upper_coeff, updates.upper_coeff),
        primitive.bind(operand.upper_bias, indices, updates.upper_bias, **kwargs),
        like,
    )


def _scatter(operand, indices, updates, **kwargs):
    return _scatter_linear(lax.scatter_p, operand, indices, updates, **kwargs)


def _scatter_add(operand, indices, updates, **kwargs):
    return _scatter_linear(lax.scatter_add_p, operand, indices, updates, **kwargs)


def _scatter_ordered(primitive, operand, indices, updates, **kwargs):
    if _is_abstract(indices):
        like = _template(operand, updates, indices)
        operand, updates = _promote(operand, like), _promote(updates, like)
        zeros = jnp.zeros_like(operand.lower_coeff)
        if primitive is lax.scatter_min_p:
            lower = jnp.minimum(operand.lower, jnp.min(updates.lower))
            return _new(
                zeros,
                lower,
                operand.upper_coeff,
                operand.upper_bias,
                like,
            )
        upper = jnp.maximum(operand.upper, jnp.max(updates.upper))
        return _new(
            operand.lower_coeff,
            operand.lower_bias,
            zeros,
            upper,
            like,
        )
    if not isinstance(operand, AffineBound) and not isinstance(updates, AffineBound):
        return primitive.bind(operand, indices, updates, **kwargs)
    like = _template(operand, updates)
    operand, updates = _promote(operand, like), _promote(updates, like)
    return _constant_interval_bounds(
        primitive.bind(operand.lower, indices, updates.lower, **kwargs),
        primitive.bind(operand.upper, indices, updates.upper, **kwargs),
        like,
    )


def _reduce_sum(x, **kwargs):
    return _unary_linear_primitive(lax.reduce_sum_p, x, **kwargs)


def _sum_affine(x: AffineBound, axes) -> AffineBound:
    return _new(
        jnp.sum(x.lower_coeff, axis=axes),
        jnp.sum(x.lower_bias, axis=axes),
        jnp.sum(x.upper_coeff, axis=axes),
        jnp.sum(x.upper_bias, axis=axes),
        x,
    )


# ---------------------------------------------------------------------------
# Cholesky decomposition and triangular solve


def _stack_affine(values, axis=0):
    """Stack affine values without mapping over their shared input domain."""
    like = _template(*values)
    values = [_promote(value, like) for value in values]
    return _new(
        jnp.stack([value.lower_coeff for value in values], axis=axis),
        jnp.stack([value.lower_bias for value in values], axis=axis),
        jnp.stack([value.upper_coeff for value in values], axis=axis),
        jnp.stack([value.upper_bias for value in values], axis=axis),
        like,
    )


def _matrix_transpose(value):
    if isinstance(value, AffineBound):
        axes = list(range(value.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        return value.transpose(tuple(axes))
    return jnp.swapaxes(value, -2, -1)


def _gershgorin_spd_lower_bound(matrix: AffineBound):
    """Sufficient uniform SPD bound from the concretized symmetric matrix."""
    lower, upper = matrix.lower, matrix.upper
    magnitude = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
    diagonal_lower = jnp.diag(lower)
    off_diagonal_sum = jnp.sum(magnitude, axis=1) - jnp.diag(magnitude)
    return jnp.min(diagonal_lower - off_diagonal_sum)


def _sqrt_with_spd_certificate(pivot: AffineBound, spd_lower):
    """Square-root relaxation rescued by a certified positive pivot bound."""
    lower, upper = pivot.lower, pivot.upper
    certified_lower = jnp.where(lower > 0, lower, spd_lower)
    valid = jnp.logical_or(lower > 0, spd_lower > 0)
    safe_lower = jnp.where(valid, certified_lower, 1.0)
    safe_upper = jnp.maximum(jnp.where(valid, upper, 1.0), safe_lower)
    width = safe_upper - safe_lower
    secant_slope = jnp.where(
        width == 0,
        1 / (2 * jnp.sqrt(safe_lower)),
        (jnp.sqrt(safe_upper) - jnp.sqrt(safe_lower)) / jnp.where(width == 0, 1, width),
    )
    tangent_point = 1 / (4 * secant_slope**2)
    result = _relax(
        pivot,
        secant_slope,
        jnp.sqrt(safe_lower) - secant_slope * safe_lower,
        secant_slope,
        jnp.sqrt(tangent_point) - secant_slope * tangent_point,
    )
    exact = _constant(jnp.sqrt(safe_lower), pivot)
    result = _select_affine(width == 0, exact, result)
    return _select_affine(
        valid, result, _constant_interval_bounds(-jnp.inf, jnp.inf, pivot)
    )


def _cholesky_recursive_2d(matrix: AffineBound):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Affine Cholesky requires a square matrix.")

    # Match immrax's natural inclusion rule and JAX's default public API.
    matrix = _scale(_add(matrix, _matrix_transpose(matrix)), 0.5)
    size = matrix.shape[0]
    spd_lower = _gershgorin_spd_lower_bound(matrix)
    zero = _constant(jnp.asarray(0, dtype=matrix.dtype), matrix)
    factor = [[zero for _ in range(size)] for _ in range(size)]

    for column in range(size):
        square_sum = zero
        for k in range(column):
            square_sum = _add(square_sum, _square(factor[column][k]))
        pivot = _sub(matrix[column, column], square_sum)
        diagonal = _sqrt_with_spd_certificate(pivot, spd_lower)
        factor[column][column] = diagonal

        for row in range(column + 1, size):
            product_sum = zero
            for k in range(column):
                product_sum = _add(product_sum, _mul(factor[row][k], factor[column][k]))
            numerator = _sub(matrix[row, column], product_sum)
            factor[row][column] = _div(numerator, diagonal)

    return _stack_affine([_stack_affine(row) for row in factor])


def _cholesky_recursive(matrix, **_kwargs):
    if not isinstance(matrix, AffineBound):
        return LA.cholesky_p.bind(matrix)
    if matrix.ndim == 2:
        return _cholesky_recursive_2d(matrix)

    batch_shape = matrix.shape[:-2]
    factors = [
        _cholesky_recursive_2d(matrix[index]) for index in np.ndindex(batch_shape)
    ]
    return _stack_affine(factors).reshape(batch_shape + matrix.shape[-2:])


def _triangular_solve_recursive_left_2d(
    matrix,
    rhs,
    *,
    lower,
    unit_diagonal,
):
    like = _template(matrix, rhs)
    matrix, rhs = _promote(matrix, like), _promote(rhs, like)
    squeeze = rhs.ndim == 1
    if squeeze:
        rhs = rhs.reshape((rhs.shape[0], 1))
    if matrix.ndim != 2 or rhs.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Baseline affine triangular solve expects a 2-D square matrix "
            "and a vector or 2-D right-hand side."
        )
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("Triangular matrix and right-hand side shapes do not match.")

    size = matrix.shape[0]
    zero_row = _constant(jnp.zeros(rhs.shape[1], dtype=rhs.dtype), like)
    solution = [zero_row for _ in range(size)]
    order = range(size) if lower else range(size - 1, -1, -1)

    for row in order:
        product_sum = zero_row
        prior = range(row) if lower else range(row + 1, size)
        for column in prior:
            product_sum = _add(product_sum, _mul(matrix[row, column], solution[column]))
        numerator = _sub(rhs[row], product_sum)
        solution[row] = (
            numerator if unit_diagonal else _div(numerator, matrix[row, row])
        )

    result = _stack_affine(solution)
    return result[:, 0] if squeeze else result


def _triangular_solve_recursive(
    matrix,
    rhs,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    del conjugate_a  # The affine domain currently has real-arithmetic semantics.
    if not isinstance(matrix, AffineBound) and not isinstance(rhs, AffineBound):
        return LA.triangular_solve_p.bind(
            matrix,
            rhs,
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=False,
            unit_diagonal=unit_diagonal,
        )

    if transpose_a:
        matrix = _matrix_transpose(matrix)
        lower = not lower
    if not left_side:
        result = _triangular_solve_recursive_left_2d(
            _matrix_transpose(matrix),
            _matrix_transpose(rhs),
            lower=not lower,
            unit_diagonal=unit_diagonal,
        )
        return _matrix_transpose(result)
    return _triangular_solve_recursive_left_2d(
        matrix, rhs, lower=lower, unit_diagonal=unit_diagonal
    )


def _affine_matmul(left, right):
    """Two-dimensional matrix product using affine/McCormick arithmetic."""
    like = _template(left, right)
    left, right = _promote(left, like), _promote(right, like)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Affine matrix multiplication requires compatible 2-D inputs.")

    zero = _constant(jnp.asarray(0, dtype=like.dtype), like)
    rows = []
    for i in range(left.shape[0]):
        row = []
        for j in range(right.shape[1]):
            value = zero
            for k in range(left.shape[1]):
                value = _add(value, _mul(left[i, k], right[k, j]))
            row.append(value)
        rows.append(_stack_affine(row))
    return _stack_affine(rows)


def _set_affine_entries(value: AffineBound, mask, constant):
    """Set selected entries to constants while retaining the shared domain."""
    mask = jnp.broadcast_to(jnp.asarray(mask), value.shape)
    constant = jnp.broadcast_to(jnp.asarray(constant, dtype=value.dtype), value.shape)
    return _new(
        jnp.where(mask[..., None], 0, value.lower_coeff),
        jnp.where(mask, constant, value.lower_bias),
        jnp.where(mask[..., None], 0, value.upper_coeff),
        jnp.where(mask, constant, value.upper_bias),
        value,
    )


def _effective_triangular_matrix(matrix: AffineBound, *, lower, unit_diagonal):
    size = matrix.shape[0]
    row, column = jnp.indices((size, size))
    unused = column > row if lower else column < row
    result = _set_affine_entries(matrix, unused, 0)
    if unit_diagonal:
        result = _set_affine_entries(result, row == column, 1)
    return result


def _interval_mul_bounds(x_lower, x_upper, y_lower, y_upper):
    products = jnp.stack(
        (
            x_lower * y_lower,
            x_lower * y_upper,
            x_upper * y_lower,
            x_upper * y_upper,
        )
    )
    # IEEE gives NaN for 0 * inf, whereas its interval extension is exactly 0.
    products = jnp.nan_to_num(products, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)
    return jnp.min(products, axis=0), jnp.max(products, axis=0)


def _interval_div_bounds(x_lower, x_upper, y_lower, y_upper):
    valid = jnp.logical_or(y_lower > 0, y_upper < 0)
    reciprocal_lower = jnp.where(valid, 1 / y_upper, -jnp.inf)
    reciprocal_upper = jnp.where(valid, 1 / y_lower, jnp.inf)
    return _interval_mul_bounds(
        x_lower, x_upper, reciprocal_lower, reciprocal_upper
    )


def _add_constant_interval(value: AffineBound, lower, upper):
    """Add a constant interval remainder to an affine predictor."""
    lower, upper = jnp.broadcast_arrays(lower, upper)
    return _new(
        value.lower_coeff,
        value.lower_bias + lower,
        value.upper_coeff,
        value.upper_bias + upper,
        value,
    )


def _triangular_remainder_interval(
    matrix: AffineBound, residual: AffineBound, *, lower, unit_diagonal
):
    """Enclose ``T R = -residual`` by a triangular interval recurrence."""
    t_lower, t_upper = matrix.lower, matrix.upper
    q_lower, q_upper = residual.lower, residual.upper
    size, columns = residual.shape
    zero = jnp.zeros((columns,), dtype=residual.dtype)
    solution_lower = [zero for _ in range(size)]
    solution_upper = [zero for _ in range(size)]
    order = range(size) if lower else range(size - 1, -1, -1)

    for row in order:
        numerator_lower = -q_upper[row]
        numerator_upper = -q_lower[row]
        prior = range(row) if lower else range(row + 1, size)
        for column in prior:
            product_lower, product_upper = _interval_mul_bounds(
                t_lower[row, column],
                t_upper[row, column],
                solution_lower[column],
                solution_upper[column],
            )
            numerator_lower = numerator_lower - product_upper
            numerator_upper = numerator_upper - product_lower
        if unit_diagonal:
            solution_lower[row], solution_upper[row] = (
                numerator_lower,
                numerator_upper,
            )
        else:
            solution_lower[row], solution_upper[row] = _interval_div_bounds(
                numerator_lower,
                numerator_upper,
                t_lower[row, row],
                t_upper[row, row],
            )
    return jnp.stack(solution_lower), jnp.stack(solution_upper)


def _triangular_solve_matrix_left_2d(
    matrix,
    rhs,
    *,
    lower,
    unit_diagonal,
):
    """Matrix-level first-order triangular solve with a certified remainder."""
    like = _template(matrix, rhs)
    matrix, rhs = _promote(matrix, like), _promote(rhs, like)
    squeeze = rhs.ndim == 1
    if squeeze:
        rhs = rhs.reshape((rhs.shape[0], 1))
    if matrix.ndim != 2 or rhs.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Matrix-level affine triangular solve expects a 2-D square matrix "
            "and a vector or 2-D right-hand side."
        )
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("Triangular matrix and right-hand side shapes do not match.")

    matrix = _effective_triangular_matrix(
        matrix, lower=lower, unit_diagonal=unit_diagonal
    )
    matrix_center = matrix.center
    rhs_center = rhs.center
    nominal = LA.triangular_solve_p.bind(
        matrix_center,
        rhs_center,
        left_side=True,
        lower=lower,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=unit_diagonal,
    )
    inverse = jnp.linalg.inv(matrix_center)
    delta_matrix = _sub(matrix, matrix_center)
    delta_rhs = _sub(rhs, rhs_center)
    first_order_rhs = _sub(
        delta_rhs, _affine_matmul(delta_matrix, nominal)
    )
    first_order = _affine_matmul(inverse, first_order_rhs)
    predictor = _add(nominal, first_order)

    quadratic_residual = _affine_matmul(delta_matrix, first_order)
    remainder_lower, remainder_upper = _triangular_remainder_interval(
        matrix,
        quadratic_residual,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    result = _add_constant_interval(predictor, remainder_lower, remainder_upper)
    return result[:, 0] if squeeze else result


def _triangular_solve_matrix(
    matrix,
    rhs,
    *,
    left_side=True,
    lower=True,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    """Matrix-level affine inclusion for ``lax.linalg.triangular_solve``."""
    del conjugate_a  # AffineBound currently represents real arithmetic.
    if not isinstance(matrix, AffineBound) and not isinstance(rhs, AffineBound):
        return LA.triangular_solve_p.bind(
            matrix,
            rhs,
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
            conjugate_a=False,
            unit_diagonal=unit_diagonal,
        )
    if transpose_a:
        matrix = _matrix_transpose(matrix)
        lower = not lower
    if not left_side:
        result = _triangular_solve_matrix_left_2d(
            _matrix_transpose(matrix),
            _matrix_transpose(rhs),
            lower=not lower,
            unit_diagonal=unit_diagonal,
        )
        return _matrix_transpose(result)
    return _triangular_solve_matrix_left_2d(
        matrix, rhs, lower=lower, unit_diagonal=unit_diagonal
    )


def _cholesky_matrix_2d(matrix: AffineBound):
    """Matrix-level Cholesky derivative plus a certified quadratic remainder."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Affine Cholesky requires a square matrix.")

    matrix = _scale(_add(matrix, _matrix_transpose(matrix)), 0.5)
    size = matrix.shape[0]
    center = matrix.center
    nominal = jnp.linalg.cholesky(center)
    inverse = jnp.linalg.inv(nominal)

    perturbation = _sub(matrix, center)
    normalized = _affine_matmul(
        _affine_matmul(inverse, perturbation), inverse.T
    )
    row, column = jnp.indices((size, size))
    phi_weight = jnp.where(row > column, 1.0, jnp.where(row == column, 0.5, 0.0))
    first_order_normalized = _scale(normalized, phi_weight)
    predictor = _add(nominal, _affine_matmul(nominal, first_order_normalized))

    magnitude = jnp.maximum(jnp.abs(normalized.lower), jnp.abs(normalized.upper))
    eta_squared = jnp.sum(jnp.where(row > column, magnitude**2, 0.0))
    eta_squared += 0.25 * jnp.sum(jnp.diag(magnitude) ** 2)
    eta = jnp.sqrt(eta_squared)
    fixed_limit = 1 / (2 * jnp.sqrt(2.0))
    fixed_valid = eta < fixed_limit
    safe_eta = jnp.where(fixed_valid, eta, 0.0)
    fixed_remainder = (
        1 - jnp.sqrt(jnp.maximum(1 - 2 * jnp.sqrt(2.0) * safe_eta, 0.0))
    ) ** 2 / (2 * jnp.sqrt(2.0))

    # ||H||_2 <= ||H||_F, so the Frobenius endpoint bound is a sound h.
    frobenius = jnp.sqrt(jnp.sum(magnitude**2))
    taylor_valid = frobenius < 1
    safe_frobenius = jnp.where(taylor_valid, frobenius, 0.0)
    taylor_remainder = (
        jnp.sqrt(1 + safe_frobenius)
        * safe_frobenius**2
        / (2 * jnp.sqrt(2.0) * (1 - safe_frobenius) ** 2)
    )
    certified_remainder = jnp.where(
        jnp.logical_and(fixed_valid, taylor_valid),
        jnp.minimum(fixed_remainder, taylor_remainder),
        jnp.where(fixed_valid, fixed_remainder, taylor_remainder),
    )
    valid = jnp.logical_and(
        jnp.logical_or(fixed_valid, taylor_valid), jnp.all(jnp.isfinite(nominal))
    )

    row_radius = jnp.sqrt(jnp.maximum(jnp.diag(center), 0)) * certified_remainder
    radius = jnp.broadcast_to(row_radius[:, None], (size, size))
    result = _add_constant_interval(predictor, -radius, radius)
    top = _constant_interval_bounds(-jnp.inf, jnp.inf, result)
    result = _select_affine(valid, result, top)
    return _set_affine_entries(result, column > row, 0)


def _cholesky_matrix(matrix, **_kwargs):
    """Experimental matrix-level affine Cholesky inclusion."""
    if not isinstance(matrix, AffineBound):
        return LA.cholesky_p.bind(matrix)
    if matrix.ndim == 2:
        return _cholesky_matrix_2d(matrix)

    batch_shape = matrix.shape[:-2]
    factors = [
        _cholesky_matrix_2d(matrix[index]) for index in np.ndindex(batch_shape)
    ]
    return _stack_affine(factors).reshape(batch_shape + matrix.shape[-2:])


def _reduce_ordered(x: AffineBound, *, is_max: bool, axes):
    if not isinstance(x, AffineBound):
        primitive = lax.reduce_max_p if is_max else lax.reduce_min_p
        if isinstance(x, Interval):
            return Interval(
                primitive.bind(x.lower, axes=axes), primitive.bind(x.upper, axes=axes)
            )
        return primitive.bind(x, axes=axes)
    axes = tuple(a % x.ndim for a in axes)
    free = tuple(i for i in range(x.ndim) if i not in axes)
    perm = free + axes
    z = x.transpose(perm)
    free_shape = tuple(x.shape[i] for i in free)
    reduction_size = int(np.prod([x.shape[i] for i in axes]))
    z = z.reshape(free_shape + (reduction_size,))
    result = z[..., 0]
    op = _maximum if is_max else _minimum
    for i in range(1, reduction_size):
        result = op(result, z[..., i])
    return result


def _reduce_max(x, *, axes):
    return _reduce_ordered(x, is_max=True, axes=axes)


def _reduce_min(x, *, axes):
    return _reduce_ordered(x, is_max=False, axes=axes)


def _reduce_bool(x, *, axes, is_and):
    primitive = lax.reduce_and_p if is_and else lax.reduce_or_p
    if isinstance(x, AffineBound):
        definitely_true = jnp.logical_or(x.lower > 0, x.upper < 0)
        possibly_true = jnp.logical_not(
            jnp.logical_and(x.lower == 0, x.upper == 0)
        )
        x = Interval(definitely_true, possibly_true)
    if not isinstance(x, Interval):
        return primitive.bind(x, axes=axes)
    return Interval(
        primitive.bind(x.lower, axes=axes), primitive.bind(x.upper, axes=axes)
    )


def _dot_general(a, b, **kwargs):
    primitive = lax.dot_general_p
    if not isinstance(a, AffineBound) and not isinstance(b, AffineBound):
        if isinstance(a, Interval) or isinstance(b, Interval):
            # Reuse the natural rule when both operands have already lost
            # affine source information, for example after a mixed predicate
            # selects between point-valued arrays.
            from . import nif

            return nif.inclusion_registry[primitive](a, b, **kwargs)
        return primitive.bind(a, b, **kwargs)
    like = _template(a, b)
    if isinstance(a, AffineBound) and not isinstance(b, (AffineBound, Interval)):
        bp, bn = jnp.maximum(b, 0), jnp.minimum(b, 0)
        dotc = lambda c, w: jax.vmap(
            lambda ci: primitive.bind(ci, w, **kwargs), in_axes=-1, out_axes=-1
        )(c)
        return _new(
            dotc(a.lower_coeff, bp) + dotc(a.upper_coeff, bn),
            primitive.bind(a.lower_bias, bp, **kwargs)
            + primitive.bind(a.upper_bias, bn, **kwargs),
            dotc(a.upper_coeff, bp) + dotc(a.lower_coeff, bn),
            primitive.bind(a.upper_bias, bp, **kwargs)
            + primitive.bind(a.lower_bias, bn, **kwargs),
            like,
        )
    if isinstance(b, AffineBound) and not isinstance(a, (AffineBound, Interval)):
        ap, an = jnp.maximum(a, 0), jnp.minimum(a, 0)
        dotc = lambda w, c: jax.vmap(
            lambda ci: primitive.bind(w, ci, **kwargs), in_axes=-1, out_axes=-1
        )(c)
        return _new(
            dotc(ap, b.lower_coeff) + dotc(an, b.upper_coeff),
            primitive.bind(ap, b.lower_bias, **kwargs)
            + primitive.bind(an, b.upper_bias, **kwargs),
            dotc(ap, b.upper_coeff) + dotc(an, b.lower_coeff),
            primitive.bind(ap, b.upper_bias, **kwargs)
            + primitive.bind(an, b.lower_bias, **kwargs),
            like,
        )

    a, b = _promote(a, like), _promote(b, like)
    (ac, bc), (ab, bb) = kwargs["dimension_numbers"]
    a = _moveaxis(a, ab + ac, tuple(range(len(ab) + len(ac))))
    b = _moveaxis(b, bb + bc, tuple(range(len(bb) + len(bc))))
    batch_n, contract_n = len(ab), len(ac)
    a_free = a.shape[batch_n + contract_n :]
    b_free = b.shape[batch_n + contract_n :]
    a = a.reshape(a.shape + (1,) * len(b_free))
    b = b.reshape(b.shape[: batch_n + contract_n] + (1,) * len(a_free) + b_free)
    products = _mul(a, b)
    contract_axes = tuple(range(batch_n, batch_n + contract_n))
    return _sum_affine(products, contract_axes) if contract_axes else products


def _moveaxis(x: AffineBound, source, destination):
    return _new(
        jnp.moveaxis(x.lower_coeff, source, destination),
        jnp.moveaxis(x.lower_bias, source, destination),
        jnp.moveaxis(x.upper_coeff, source, destination),
        jnp.moveaxis(x.upper_bias, source, destination),
        x,
    )


affine_inclusion_registry[lax.copy_p] = _copy
affine_inclusion_registry[lax.stop_gradient_p] = _stop_gradient
affine_inclusion_registry[lax.reshape_p] = _reshape
affine_inclusion_registry[lax.slice_p] = _slice
affine_inclusion_registry[lax.squeeze_p] = _squeeze
affine_inclusion_registry[lax.transpose_p] = _transpose
affine_inclusion_registry[lax.broadcast_in_dim_p] = _broadcast_in_dim
if hasattr(lax, "rev_p"):
    affine_inclusion_registry[lax.rev_p] = lambda x, **kwargs: _unary_linear_primitive(
        lax.rev_p, x, **kwargs
    )
affine_inclusion_registry[lax.split_p] = _split
affine_inclusion_registry[lax.concatenate_p] = _concatenate
affine_inclusion_registry[lax.dynamic_slice_p] = _dynamic_slice
affine_inclusion_registry[lax.gather_p] = _gather
affine_inclusion_registry[lax.pad_p] = _pad
affine_inclusion_registry[lax.scatter_p] = _scatter
affine_inclusion_registry[lax.scatter_add_p] = _scatter_add
affine_inclusion_registry[lax.scatter_min_p] = lambda *a, **kw: _scatter_ordered(
    lax.scatter_min_p, *a, **kw
)
affine_inclusion_registry[lax.scatter_max_p] = lambda *a, **kw: _scatter_ordered(
    lax.scatter_max_p, *a, **kw
)
affine_inclusion_registry[lax.reduce_sum_p] = _reduce_sum
affine_inclusion_registry[lax.reduce_max_p] = _reduce_max
affine_inclusion_registry[lax.reduce_min_p] = _reduce_min
affine_inclusion_registry[lax.reduce_and_p] = lambda x, *, axes: _reduce_bool(
    x, axes=axes, is_and=True
)
affine_inclusion_registry[lax.reduce_or_p] = lambda x, *, axes: _reduce_bool(
    x, axes=axes, is_and=False
)
affine_inclusion_registry[lax.dot_general_p] = _dot_general

# Python ``@`` can be evaluated directly inside a custom affine inclusion
# rule, before a surrounding Jaxpr has a chance to lower it to ``dot_general``.
# Route that operator through the same registered affine interpreter used by
# ordinary traced ``jnp.matmul`` calls.
_matmul_affine = affif(jnp.matmul, return_type="affine")
AffineBound.__matmul__ = _matmul_affine
AffineBound.__rmatmul__ = lambda self, other: _matmul_affine(other, self)

# The recursive enclosures are tighter on the regression boxes below.  Keep
# the matrix-level implementations available for experimentation, but do not
# register them as the public ``affif`` defaults.
affine_inclusion_registry[LA.cholesky_p] = _cholesky_recursive
affine_inclusion_registry[LA.triangular_solve_p] = _triangular_solve_recursive
# ``iota`` has no data operand and is normally evaluated directly by the
# interpreter; registering it documents parity with nif and supports direct
# registry use.
affine_inclusion_registry[lax.iota_p] = lambda **kwargs: lax.iota_p.bind(**kwargs)


def _atan2(y, x):
    if not isinstance(x, AffineBound) and not isinstance(y, AffineBound):
        return lax.atan2(y, x)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    right = _atan(_div(y, x))
    upper_half = _sub(jnp.pi / 2, _atan(_div(x, y)))
    lower_half = _sub(-jnp.pi / 2, _atan(_div(x, y)))
    negative_x_nonnegative_y = _add(_atan(_div(y, x)), jnp.pi)
    full_angle = _constant_interval_bounds(-jnp.pi, jnp.pi, like)
    result = full_angle
    result = _select_affine(
        jnp.logical_and(x.upper < 0, y.lower >= 0), negative_x_nonnegative_y, result
    )
    result = _select_affine(y.upper < 0, lower_half, result)
    result = _select_affine(y.lower > 0, upper_half, result)
    result = _select_affine(x.lower > 0, right, result)
    return _jointly_degenerate_exact(result, (y, x), jnp.arctan2)


if hasattr(lax, "clamp_p"):
    affine_inclusion_registry[lax.clamp_p] = _clamp
affine_inclusion_registry[lax.atan2_p] = _atan2


# ---------------------------------------------------------------------------
# Higher-order primitives


def _jit_primitive(*args, **bind_params):
    closed = bind_params.pop("jaxpr")
    if isinstance(closed, jax.extend.core.ClosedJaxpr):
        return aif_jaxpr(closed.jaxpr, closed.consts, *args)
    return aif_jaxpr(closed, [], *args)


def _stack_values(values):
    first = values[0]
    if isinstance(first, AffineBound):
        return _new(
            jnp.stack([v.lower_coeff for v in values]),
            jnp.stack([v.lower_bias for v in values]),
            jnp.stack([v.upper_coeff for v in values]),
            jnp.stack([v.upper_bias for v in values]),
            first,
        )
    if isinstance(first, Interval):
        return Interval(
            jnp.stack([v.lower for v in values]), jnp.stack([v.upper for v in values])
        )
    return jnp.stack(values)


def _scan(*args, **bind_params):
    closed = bind_params["jaxpr"]
    if isinstance(closed, jax.extend.core.ClosedJaxpr):
        embedded_consts = list(closed.consts)
        body = closed.jaxpr
    else:
        embedded_consts, body = [], closed
    num_consts = bind_params["num_consts"]
    num_carry = bind_params["num_carry"]
    length = bind_params["length"]
    reverse = bind_params.get("reverse", False)
    constants = embedded_consts or list(args[:num_consts])
    carry = list(args[num_consts : num_consts + num_carry])
    xs = list(args[num_consts + num_carry :])
    collected = None
    indices = range(length - 1, -1, -1) if reverse else range(length)
    for i in indices:
        step_x = [x[i] for x in xs]
        outputs = list(aif_jaxpr(body, [], *constants, *carry, *step_x))
        carry, step_y = outputs[:num_carry], outputs[num_carry:]
        if collected is None:
            collected = [[] for _ in step_y]
        for bucket, value in zip(collected, step_y):
            bucket.append(value)
    if not collected:
        ys = []
    else:
        if reverse:
            collected = [list(reversed(bucket)) for bucket in collected]
        ys = [_stack_values(bucket) for bucket in collected]
    return [*carry, *ys]


def _union_values(*values):
    if any(isinstance(v, AffineBound) for v in values):
        like = _template(*values)
        values = [_promote(v, like) for v in values]
        lower = jnp.min(jnp.stack([v.lower for v in values]), axis=0)
        upper = jnp.max(jnp.stack([v.upper for v in values]), axis=0)
        return _constant_interval_bounds(lower, upper, like)
    if any(isinstance(v, Interval) for v in values):
        values = [interval(v) for v in values]
        return Interval(
            jnp.min(jnp.stack([v.lower for v in values]), axis=0),
            jnp.max(jnp.stack([v.upper for v in values]), axis=0),
        )
    return values[0]


def _cond(index, *args, **bind_params):
    if isinstance(index, AffineBound):
        index = index.concretize()
    branches = bind_params["branches"]
    branch_outputs = []
    for branch in branches:
        if isinstance(branch, jax.extend.core.ClosedJaxpr):
            branch_outputs.append(aif_jaxpr(branch.jaxpr, branch.consts, *args))
        else:
            branch_outputs.append(aif_jaxpr(branch, [], *args))
    if isinstance(index, Interval):
        if index.lower.dtype == jnp.bool_ and len(branch_outputs) == 2:
            return [
                _select(index, true_value, false_value)
                for false_value, true_value in zip(*branch_outputs)
            ]
        return [_select_n(index, *group) for group in zip(*branch_outputs)]
    return [
        lax.select_n(index, *group)
        if not any(_is_abstract(v) for v in group)
        else _select_n(Interval(index, index), *group)
        for group in zip(*branch_outputs)
    ]


def _while(*args, **bind_params):
    """Evaluate a while primitive over its affine carry pytree.

    This directly supports the report's common finitely bounded form, where a
    concrete loop counter supplies the termination condition.  As with natif,
    the lower truth value is used when a predicate itself is interval-valued;
    callers should express uncertain termination as an explicitly bounded
    scan/conditional so all possible stopping times can be joined.
    """
    cond_closed = bind_params["cond_jaxpr"]
    body_closed = bind_params["body_jaxpr"]
    cond_body = (
        cond_closed.jaxpr
        if isinstance(cond_closed, jax.extend.core.ClosedJaxpr)
        else cond_closed
    )
    body = (
        body_closed.jaxpr
        if isinstance(body_closed, jax.extend.core.ClosedJaxpr)
        else body_closed
    )
    cond_embedded = (
        list(cond_closed.consts)
        if isinstance(cond_closed, jax.extend.core.ClosedJaxpr)
        else []
    )
    body_embedded = (
        list(body_closed.consts)
        if isinstance(body_closed, jax.extend.core.ClosedJaxpr)
        else []
    )
    nc, nb = bind_params["cond_nconsts"], bind_params["body_nconsts"]
    cond_consts = cond_embedded or list(args[:nc])
    body_consts = body_embedded or list(args[nc : nc + nb])
    carry = list(args[nc + nb :])
    if any(isinstance(v, AffineBound) for v in (*body_consts, *carry)):
        like = _template(*body_consts, *carry)
        carry = [_promote(v, like) for v in carry]

    def cond_function(loop_carry):
        pred = aif_jaxpr(cond_body, [], *cond_consts, *loop_carry)[0]
        return pred.lower if isinstance(pred, Interval) else pred

    def body_function(loop_carry):
        return tuple(aif_jaxpr(body, [], *body_consts, *loop_carry))

    return list(lax.while_loop(cond_function, body_function, tuple(carry)))


try:
    affine_inclusion_registry[jax._src.pjit.jit_p] = _jit_primitive
except AttributeError:
    pass
affine_inclusion_registry[lax.scan_p] = _scan
affine_inclusion_registry[lax.cond_p] = _cond
affine_inclusion_registry[lax.while_p] = _while


def _debug_callback(*args, **kwargs):
    lowered = [a.concretize() if isinstance(a, AffineBound) else a for a in args]
    return debug_callback_p.bind(*lowered, **kwargs)


affine_inclusion_registry[debug_callback_p] = _debug_callback


__all__ = ["affif", "affine_inclusion_registry", "aif_jaxpr"]
