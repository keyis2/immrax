"""Experimental source-dependent quadratic endpoint inclusion transform.

A PairedQuadratic represents [L2(xi), U2(xi)] on one shared normalized
source box. L2 and U2 are degree-2 endpoint polynomials stored as zero-
remainder TaylorModel containers. The concrete value lies between their
values at *every fixed* xi. This module is opt-in and leaves tmif unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import ad_util
from jax.tree_util import register_pytree_node_class

from .interval import Interval
from .taylor import (
    TaylorModel,
    _model_is_finite,
    _nan_model_like,
    _where_model,
    constant_taylor_model,
    evaluate_polynomial,
    interpret_inclusion_jaxpr,
    normalized_taylor_seed,
    polynomial_range,
    taylor_endpoint_models,
    taylor_inclusion_registry,
)


@register_pytree_node_class
class PairedQuadratic:
    """Two ordered degree-2 endpoint polynomials over shared sources.

    ``valid`` propagates numerical/domain preconditions. Static shape and
    dtype requirements are checked at construction. Use
    :func:`pair_status` for the certified ordering and finite predicates.
    """

    def __init__(self, lower: TaylorModel, upper: TaylorModel, valid=True):
        if not isinstance(lower, TaylorModel) or not isinstance(upper, TaylorModel):
            raise TypeError("paired endpoints must be TaylorModel polynomial containers")
        if lower.shape != upper.shape or lower.source_size != upper.source_size:
            raise ValueError("paired endpoints must have equal value/source shapes")
        if lower.dtype != upper.dtype or not jnp.issubdtype(lower.dtype, jnp.floating):
            raise ValueError("paired endpoints must have the same floating dtype")
        self.lower = lower
        self.upper = upper
        self.valid = jnp.broadcast_to(jnp.asarray(valid), lower.shape)

    def tree_flatten(self):
        return ((self.lower, self.upper, self.valid), None)

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def shape(self):
        return self.lower.shape

    @property
    def ndim(self):
        return self.lower.ndim

    @property
    def source_size(self):
        return self.lower.source_size

    @property
    def dtype(self):
        return self.lower.dtype

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        return PairedQuadratic(self.lower[index], self.upper[index], self.valid[index])

    def reshape(self, *shape):
        return PairedQuadratic(self.lower.reshape(*shape), self.upper.reshape(*shape), self.valid.reshape(*shape))

    def transpose(self, *axes):
        return PairedQuadratic(self.lower.transpose(*axes), self.upper.transpose(*axes), self.valid.transpose(*axes))

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(other, self)

    def __neg__(self):
        return _neg(self)

    def __sub__(self, other):
        return _sub(self, other)

    def __rsub__(self, other):
        return _sub(other, self)

    def __mul__(self, other):
        return _mul(self, other)

    def __rmul__(self, other):
        return _mul(other, self)

    def __truediv__(self, other):
        return _div(self, other)

    def __rtruediv__(self, other):
        return _div(other, self)


def pair_status(value: PairedQuadratic) -> dict[str, jax.Array]:
    """Return sound polynomial-order and finite-value checks by component."""

    zero_remainder = (
        (value.lower.remainder.lower == 0)
        & (value.lower.remainder.upper == 0)
        & (value.upper.remainder.lower == 0)
        & (value.upper.remainder.upper == 0)
    )
    ordering_margin = polynomial_range(value.upper - value.lower).lower
    finite = _model_is_finite(value.lower) & _model_is_finite(value.upper)
    finite &= jnp.all(jnp.isfinite(ordering_margin))
    return {
        "valid": value.valid,
        "zero_remainder": zero_remainder,
        "ordering_margin": ordering_margin,
        "ordered": ordering_margin >= 0,
        "finite": finite,
    }


def pair_range(value: PairedQuadratic) -> Interval:
    """Sound global range from the two endpoint polynomials."""

    return Interval(polynomial_range(value.lower).lower, polynomial_range(value.upper).upper)


def certify_pair_order(value: PairedQuadratic) -> PairedQuadratic:
    """Raise the upper endpoint enough for termwise ordering certification.

    Call only after an analytic inclusion rule has established that both raw
    endpoints enclose the concrete value. The adjustment uses the sound
    polynomial range of U2-L2, never sampled values. It may widen the pair.
    """

    margin = polynomial_range(value.upper - value.lower).lower
    shift = jnp.maximum(-margin, 0.0)
    upper = value.upper + shift
    return PairedQuadratic(value.lower, upper, value.valid & jnp.isfinite(shift))


def evaluate_pair(value: PairedQuadratic, sources) -> tuple[jax.Array, jax.Array]:
    """Evaluate both endpoint polynomials at matching normalized sources."""

    return evaluate_polynomial(value.lower, sources), evaluate_polynomial(value.upper, sources)


def constant_pair(value, source_size) -> PairedQuadratic:
    model = constant_taylor_model(value, source_size)
    return PairedQuadratic(model, model)


def normalized_pair_seed(lower, upper) -> PairedQuadratic:
    model = normalized_taylor_seed(lower, upper)
    return PairedQuadratic(model, model)


def pair_from_taylor(model: TaylorModel) -> PairedQuadratic:
    lower, upper = taylor_endpoint_models(model)
    return PairedQuadratic(lower, upper)


def pair_to_taylor(value: PairedQuadratic, *, cause="direct") -> TaylorModel:
    """Sound explicit pair -> M2 + [-rho,rho] Taylor conversion.

    M2=(L2+U2)/2; D2=(U2-L2)/2; rho bounds |D2| on the full
    normalized box. Invalid ordering or nonfinite coefficients produce NaNs.
    """

    midpoint = 0.5 * (value.lower + value.upper)
    half_width = 0.5 * (value.upper - value.lower)
    bounds = polynomial_range(half_width)
    radius = jnp.maximum(jnp.abs(bounds.lower), jnp.abs(bounds.upper))
    status = pair_status(value)
    valid = value.valid & jnp.all(status["zero_remainder"])
    valid &= jnp.all(status["ordered"]) & status["finite"]
    valid &= jnp.all(jnp.isfinite(radius))
    record_pair_predicate(f"pair_to_taylor.{cause}.input_valid", value.valid)
    record_pair_predicate(f"pair_to_taylor.{cause}.ordered", status["ordered"])
    record_pair_predicate(f"pair_to_taylor.{cause}.finite", status["finite"])
    record_pair_predicate(f"pair_to_taylor.{cause}.order_finite", valid)
    converted = TaylorModel(
        midpoint.constant, midpoint.linear, midpoint.quadratic,
        Interval(-radius, radius),
    )
    return _where_model(valid, converted, _nan_model_like(converted))


def _template(*values) -> PairedQuadratic:
    for value in values:
        if isinstance(value, PairedQuadratic):
            return value
    raise TypeError("paired operation needs a PairedQuadratic operand")


def _promote(value, like: PairedQuadratic) -> PairedQuadratic:
    if isinstance(value, PairedQuadratic):
        if value.source_size != like.source_size:
            raise ValueError("paired operands have different source dimensions")
        return value
    if isinstance(value, Interval):
        raise TypeError("interval promotion is outside the paired contract")
    return constant_pair(value, like.source_size)


def _select(mask, yes: PairedQuadratic, no: PairedQuadratic) -> PairedQuadratic:
    return PairedQuadratic(
        _where_model(mask, yes.lower, no.lower),
        _where_model(mask, yes.upper, no.upper),
        jnp.where(mask, yes.valid, no.valid),
    )


def _add(x, y):
    if (isinstance(x, PairedQuadratic) or isinstance(y, PairedQuadratic)) and (
        isinstance(x, Interval) or isinstance(y, Interval)
    ):
        return _fallback(lax.add_p, x, y)
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[lax.add_p](x, y)
        return lax.add_p.bind(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    return PairedQuadratic(x.lower + y.lower, x.upper + y.upper, x.valid & y.valid)


def _neg(x):
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[lax.neg_p](x)
        return lax.neg_p.bind(x)
    return PairedQuadratic(-x.upper, -x.lower, x.valid)


def _sub(x, y):
    return _add(x, _neg(y))


def _scale(x: PairedQuadratic, scalar) -> PairedQuadratic:
    scalar = jnp.asarray(scalar)
    nonnegative = scalar >= 0
    return PairedQuadratic(
        _where_model(nonnegative, x.lower * scalar, x.upper * scalar),
        _where_model(nonnegative, x.upper * scalar, x.lower * scalar),
        x.valid & jnp.all(jnp.isfinite(scalar)),
    )


class PairAudit:
    """Static interpretation ledger and runtime gates for hybrid products."""

    def __init__(self):
        self.calls: list[tuple[str, str, Any]] = []

    def add(self, primitive, mode, predicate=None):
        self.calls.append((primitive.name if hasattr(primitive, "name") else primitive, mode, predicate))


_active_audit: ContextVar[PairAudit | None] = ContextVar("paired_quadratic_audit", default=None)


def _record(primitive, mode, predicate=None):
    audit = _active_audit.get()
    if audit is not None:
        audit.add(primitive, mode, predicate)


def record_pair_predicate(name, predicate):
    """Record a numerical rule precondition in the active diagnostic audit."""
    if isinstance(predicate, jax.core.Tracer):
        return  # The enclosing vmap records its aggregate result after batching.
    _record(name, "predicate", predicate)


def _nonnegative_product(x: PairedQuadratic, y: PairedQuadratic):
    # The monotone endpoint products have zero-remainder quadratic endpoints.
    # Taylor's generic product bounds discarded degree 3/4 terms with fixed-
    # rank O(n^2) interval overflow. No Python monomial enumeration occurs.
    lower, _ = taylor_endpoint_models(x.lower * y.lower)
    _, upper = taylor_endpoint_models(x.upper * y.upper)
    return PairedQuadratic(lower, upper, x.valid & y.valid)


def _fallback(primitive, *args, **params):
    if primitive not in _allowed_fallback_primitives:
        raise NotImplementedError(
            f"paired Taylor fallback was not explicitly registered: {primitive.name}"
        )
    rule = taylor_inclusion_registry[primitive]
    converted = tuple(pair_to_taylor(x, cause=primitive.name) if isinstance(x, PairedQuadratic) else x for x in args)
    answer = rule(*converted, **params)
    if primitive not in _fallback_primitives:
        _record(primitive, "Taylor fallback")
    if isinstance(answer, TaylorModel):
        return pair_from_taylor(answer)
    if isinstance(answer, list):
        return [pair_from_taylor(x) if isinstance(x, TaylorModel) else x for x in answer]
    if isinstance(answer, tuple):
        return tuple(pair_from_taylor(x) if isinstance(x, TaylorModel) else x for x in answer)
    return answer


def register_pair_taylor_fallback(primitive):
    """Opt one existing Taylor primitive rule into explicit pair conversion."""

    if primitive not in taylor_inclusion_registry:
        raise ValueError(f"Taylor has no rule for {primitive.name}")
    pair_inclusion_registry[primitive] = lambda *args, **params: _fallback(primitive, *args, **params)
    _fallback_primitives.add(primitive)
    _allowed_fallback_primitives.add(primitive)


def _mul(x, y):
    if (isinstance(x, PairedQuadratic) or isinstance(y, PairedQuadratic)) and (
        isinstance(x, Interval) or isinstance(y, Interval)
    ):
        return _fallback(lax.mul_p, x, y)
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[lax.mul_p](x, y)
        return lax.mul_p.bind(x, y)
    if isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        return _scale(x, y)
    if isinstance(y, PairedQuadratic) and not isinstance(x, PairedQuadratic):
        return _scale(y, x)
    x, y = _promote(x, y), _promote(y, x)
    x_status, y_status = pair_status(x), pair_status(y)
    nonnegative = (
        x.valid & y.valid
        & jnp.all(x_status["ordered"]) & jnp.all(y_status["ordered"])
        & jnp.all(x_status["zero_remainder"]) & jnp.all(y_status["zero_remainder"])
        & (pair_range(x).lower >= 0) & (pair_range(y).lower >= 0)
    )
    native = _nonnegative_product(x, y)
    fallback = pair_from_taylor(
        pair_to_taylor(x, cause="mul.left") * pair_to_taylor(y, cause="mul.right")
    )
    _record(lax.mul_p, "native nonnegative / Taylor fallback", nonnegative)
    return _select(nonnegative, native, fallback)


def _div(x, y):
    if (isinstance(x, PairedQuadratic) or isinstance(y, PairedQuadratic)) and (
        isinstance(x, Interval) or isinstance(y, Interval)
    ):
        return _fallback(lax.div_p, x, y)
    if isinstance(y, PairedQuadratic):
        return _fallback(lax.div_p, x, y)
    if isinstance(x, PairedQuadratic):
        return _scale(x, jnp.reciprocal(jnp.asarray(y)))
    if isinstance(x, Interval) or isinstance(y, Interval):
        return taylor_inclusion_registry[lax.div_p](x, y)
    return lax.div_p.bind(x, y)


def _linear_unary(primitive, x, **params):
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[primitive](x, **params)
        return primitive.bind(x, **params)
    rule = taylor_inclusion_registry[primitive]
    valid = x.valid if primitive is lax.convert_element_type_p else primitive.bind(x.valid, **params)
    return PairedQuadratic(rule(x.lower, **params), rule(x.upper, **params), valid)


def _concatenate(*args, **params):
    if any(isinstance(x, PairedQuadratic) for x in args) and any(isinstance(x, Interval) for x in args):
        return _fallback(lax.concatenate_p, *args, **params)
    if not any(isinstance(x, PairedQuadratic) for x in args):
        if any(isinstance(x, Interval) for x in args):
            return taylor_inclusion_registry[lax.concatenate_p](*args, **params)
        return lax.concatenate_p.bind(*args, **params)
    like = _template(*args)
    values = [_promote(x, like) for x in args]
    rule = taylor_inclusion_registry[lax.concatenate_p]
    return PairedQuadratic(
        rule(*(x.lower for x in values), **params),
        rule(*(x.upper for x in values), **params),
        jnp.concatenate([x.valid for x in values], axis=params["dimension"]),
    )


def _gather(x, indices, **params):
    if isinstance(indices, (PairedQuadratic, Interval)):
        raise NotImplementedError("paired gather requires point-valued indices")
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[lax.gather_p](x, indices, **params)
        return lax.gather_p.bind(x, indices, **params)
    rule = taylor_inclusion_registry[lax.gather_p]
    return PairedQuadratic(
        rule(x.lower, indices, **params), rule(x.upper, indices, **params),
        lax.gather_p.bind(x.valid, indices, **params),
    )


def _scatter(primitive, operand, indices, updates, **params):
    if (isinstance(operand, PairedQuadratic) or isinstance(updates, PairedQuadratic)) and (
        isinstance(operand, Interval) or isinstance(updates, Interval)
    ):
        return _fallback(primitive, operand, indices, updates, **params)
    if isinstance(indices, (PairedQuadratic, Interval)):
        raise NotImplementedError("paired scatter requires point-valued indices")
    if not isinstance(operand, PairedQuadratic) and not isinstance(updates, PairedQuadratic):
        if isinstance(operand, Interval) or isinstance(updates, Interval):
            return taylor_inclusion_registry[primitive](operand, indices, updates, **params)
        return primitive.bind(operand, indices, updates, **params)
    like = _template(operand, updates)
    operand, updates = _promote(operand, like), _promote(updates, like)
    rule = taylor_inclusion_registry[primitive]
    return PairedQuadratic(
        rule(operand.lower, indices, updates.lower, **params),
        rule(operand.upper, indices, updates.upper, **params),
        jnp.all(operand.valid) & jnp.all(updates.valid),
    )


def _reduce_sum(x, *, axes, **params):
    if not isinstance(x, PairedQuadratic):
        return _linear_unary(lax.reduce_sum_p, x, axes=axes, **params)
    rule = taylor_inclusion_registry[lax.reduce_sum_p]
    return PairedQuadratic(
        rule(x.lower, axes=axes, **params),
        rule(x.upper, axes=axes, **params),
        jnp.all(x.valid, axis=axes),
    )


def _moveaxis(x: PairedQuadratic, source, destination):
    from .taylor import _moveaxis as taylor_moveaxis
    return PairedQuadratic(
        taylor_moveaxis(x.lower, source, destination),
        taylor_moveaxis(x.upper, source, destination),
        jnp.moveaxis(x.valid, source, destination),
    )


def _dot_general(a, b, **params):
    if (isinstance(a, PairedQuadratic) or isinstance(b, PairedQuadratic)) and (
        isinstance(a, Interval) or isinstance(b, Interval)
    ):
        return _fallback(lax.dot_general_p, a, b, **params)
    if not isinstance(a, PairedQuadratic) and not isinstance(b, PairedQuadratic):
        return lax.dot_general_p.bind(a, b, **params)
    like = _template(a, b)
    a, b = _promote(a, like), _promote(b, like)
    (ac, bc), (ab, bb) = params["dimension_numbers"]
    a = _moveaxis(a, ab + ac, tuple(range(len(ab) + len(ac))))
    b = _moveaxis(b, bb + bc, tuple(range(len(bb) + len(bc))))
    batch_n, contract_n = len(ab), len(ac)
    a_free = a.shape[batch_n + contract_n:]
    b_free = b.shape[batch_n + contract_n:]
    a = a.reshape(a.shape + (1,) * len(b_free))
    b = b.reshape(b.shape[:batch_n + contract_n] + (1,) * len(a_free) + b_free)
    products = _mul(a, b)
    axes = tuple(range(batch_n, batch_n + contract_n))
    return _reduce_sum(products, axes=axes) if axes else products


def _comparison(kind, x, y):
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[getattr(lax, kind + "_p")](x, y)
        return getattr(lax, kind + "_p").bind(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    diff = PairedQuadratic(x.lower - y.upper, x.upper - y.lower, x.valid & y.valid)
    bounds = pair_range(diff)
    if kind == "lt":
        return Interval(bounds.upper < 0, bounds.lower < 0)
    if kind == "le":
        return Interval(bounds.upper <= 0, bounds.lower <= 0)
    if kind == "gt":
        return _comparison("lt", y, x)
    if kind == "ge":
        return _comparison("le", y, x)
    if kind == "eq":
        return Interval((bounds.lower == 0) & (bounds.upper == 0),
                        (bounds.lower <= 0) & (bounds.upper >= 0))
    if kind == "ne":
        eq = _comparison("eq", x, y)
        return Interval(~eq.upper, ~eq.lower)
    raise ValueError(kind)


def _select_n(which, *cases):
    if isinstance(which, Interval):
        return _fallback(lax.select_n_p, which, *cases)
    if not any(isinstance(x, PairedQuadratic) for x in cases):
        return lax.select_n(which, *cases)
    if any(isinstance(x, Interval) for x in cases):
        return _fallback(lax.select_n_p, which, *cases)
    like = _template(*cases)
    values = [_promote(x, like) for x in cases]
    # select_n on coefficient containers goes through the already tested
    # Taylor fixed-index selector; arity is fixed by the traced primitive.
    rule = taylor_inclusion_registry[lax.select_n_p]
    lower = rule(which, *(x.lower for x in values))
    upper = rule(which, *(x.upper for x in values))
    return PairedQuadratic(lower, upper, lax.select_n_p.bind(which, *(x.valid for x in values)))


def _split(x, **params):
    if not isinstance(x, PairedQuadratic):
        return lax.split_p.bind(x, **params)
    rule = taylor_inclusion_registry[lax.split_p]
    lowers, uppers = rule(x.lower, **params), rule(x.upper, **params)
    valids = lax.split_p.bind(x.valid, **params)
    return [PairedQuadratic(l, u, v) for l, u, v in zip(lowers, uppers, valids)]


def _convert(x, **params):
    if not isinstance(x, PairedQuadratic):
        return taylor_inclusion_registry[lax.convert_element_type_p](x, **params)
    dtype = params["new_dtype"]
    if dtype == jnp.bool_:
        bounds = pair_range(x)
        return Interval((bounds.lower > 0) | (bounds.upper < 0),
                        ~((bounds.lower == 0) & (bounds.upper == 0)))
    if not jnp.issubdtype(dtype, jnp.floating):
        raise NotImplementedError("paired conversion to nonfloating value unsupported")
    return _linear_unary(lax.convert_element_type_p, x, **params)


pair_inclusion_registry: dict[Any, Callable[..., Any]] = {
    lax.add_p: _add,
    ad_util.add_any_p: _add,
    lax.sub_p: _sub,
    lax.neg_p: _neg,
    lax.mul_p: _mul,
    lax.div_p: _div,
    lax.reshape_p: lambda x, **p: _linear_unary(lax.reshape_p, x, **p),
    lax.slice_p: lambda x, **p: _linear_unary(lax.slice_p, x, **p),
    lax.squeeze_p: lambda x, **p: _linear_unary(lax.squeeze_p, x, **p),
    lax.transpose_p: lambda x, **p: _linear_unary(lax.transpose_p, x, **p),
    lax.broadcast_in_dim_p: lambda x, **p: _linear_unary(lax.broadcast_in_dim_p, x, **p),
    lax.concatenate_p: _concatenate,
    lax.gather_p: _gather,
    lax.scatter_p: lambda *args, **p: _scatter(lax.scatter_p, *args, **p),
    lax.scatter_add_p: lambda *args, **p: _scatter(lax.scatter_add_p, *args, **p),
    lax.reduce_sum_p: _reduce_sum,
    lax.reduce_and_p: taylor_inclusion_registry[lax.reduce_and_p],
    lax.reduce_or_p: taylor_inclusion_registry[lax.reduce_or_p],
    lax.and_p: taylor_inclusion_registry[lax.and_p],
    lax.or_p: taylor_inclusion_registry[lax.or_p],
    lax.not_p: taylor_inclusion_registry[lax.not_p],
    lax.lt_p: lambda x, y: _comparison("lt", x, y),
    lax.le_p: lambda x, y: _comparison("le", x, y),
    lax.gt_p: lambda x, y: _comparison("gt", x, y),
    lax.ge_p: lambda x, y: _comparison("ge", x, y),
    lax.eq_p: lambda x, y: _comparison("eq", x, y),
    lax.ne_p: lambda x, y: _comparison("ne", x, y),
    lax.select_n_p: _select_n,
    lax.convert_element_type_p: _convert,
    lax.copy_p: lambda x: _linear_unary(lax.copy_p, x),
    lax.stop_gradient_p: lambda x: _linear_unary(lax.stop_gradient_p, x),
    lax.dot_general_p: _dot_general,
}

if hasattr(lax, "split_p"):
    pair_inclusion_registry[lax.split_p] = _split

_allowed_fallback_primitives: set[Any] = {
    lax.add_p, lax.mul_p, lax.div_p, lax.dot_general_p,
    lax.concatenate_p, lax.select_n_p, lax.scatter_p, lax.scatter_add_p,
}
_fallback_primitives: set[Any] = set()
for _primitive in (lax.sin_p, lax.cos_p, lax.max_p, lax.min_p, lax.abs_p, lax.sqrt_p):
    register_pair_taylor_fallback(_primitive)


def register_pair_rule(primitive, rule: Callable[..., Any]):
    """Register a primitive owner rule in the opt-in paired interpreter."""
    pair_inclusion_registry[primitive] = rule


def _pair_jit_rule(*args, **params):
    closed = params.pop("jaxpr")
    if isinstance(closed, jax.extend.core.ClosedJaxpr):
        return pair_jaxpr(closed.jaxpr, closed.consts, *args)
    return pair_jaxpr(closed, [], *args)

try:
    pair_inclusion_registry[jax._src.pjit.jit_p] = _pair_jit_rule
except AttributeError:  # pragma: no cover
    pass


def _is_abstract(value):
    return isinstance(value, (PairedQuadratic, Interval))


def pair_jaxpr(jaxpr, consts, *args):
    audit = _active_audit.get()
    return interpret_inclusion_jaxpr(
        jaxpr, consts, *args,
        registry=pair_inclusion_registry,
        is_abstract=_is_abstract,
        label="paired quadratic",
        on_rule=(lambda primitive: audit.add(primitive, "Taylor fallback" if primitive in _fallback_primitives else "native")) if audit is not None else None,
    )


def pqif(function: Callable[..., Any]) -> Callable[..., Any]:
    """Transform a point function through ordered quadratic endpoint pairs."""

    @wraps(function)
    def wrapped(*args, audit: PairAudit | None = None, **kwargs):
        leaves = jax.tree_util.tree_leaves(args, is_leaf=_is_abstract)
        abstract = [x for x in leaves if isinstance(x, PairedQuadratic)]
        if not abstract:
            return function(*args, **kwargs)
        source_size = abstract[0].source_size
        if any(x.source_size != source_size for x in abstract[1:]):
            raise ValueError("paired operands must share one source dimension")
        trace_args = jax.tree_util.tree_map(
            lambda x: x.lower.constant if isinstance(x, PairedQuadratic) else x,
            args, is_leaf=_is_abstract,
        )
        trace_kwargs = jax.tree_util.tree_map(
            lambda x: x.lower.constant if isinstance(x, PairedQuadratic) else x,
            kwargs, is_leaf=_is_abstract,
        )
        closed = eqx.filter_make_jaxpr(lambda *pos: function(*pos, **trace_kwargs))(*trace_args)[0]
        flat_args = [x for x in leaves if isinstance(x, PairedQuadratic) or eqx.is_array(x)]
        token = _active_audit.set(audit)
        try:
            outputs = pair_jaxpr(closed.jaxpr, closed.literals, *flat_args)
        finally:
            _active_audit.reset(token)
        return outputs[0] if len(outputs) == 1 else outputs

    return wrapped
