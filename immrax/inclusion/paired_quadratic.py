"""Experimental source-dependent quadratic endpoint inclusion transform.

A PairedQuadratic represents [L2(xi), U2(xi)] on one shared normalized
source box. L2 and U2 are degree-2 endpoint polynomials stored as zero-
remainder TaylorModel containers. The concrete value lies between their
values at *every fixed* xi. This module is opt-in and leaves tmif unchanged.
Pair multiplication defaults to the native midpoint/half-width rule;
``four_candidate`` and the legacy ``taylor`` strategy remain explicit options.
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
    _broadcast_model,
    _model_is_finite,
    _nan_model_like,
    _square as _taylor_square,
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


def pair_from_interval(value: Interval, source_size) -> PairedQuadratic:
    """Embed one floating numeric interval as exact constant endpoints.

    Boolean and integer intervals encode control or candidate information and
    are deliberately not interpreted as continuous paired values.
    """
    if not isinstance(value, Interval):
        raise TypeError("pair_from_interval requires an Interval")
    if value.lower.shape != value.upper.shape or value.lower.dtype != value.upper.dtype:
        raise ValueError("numeric interval endpoints must share shape and dtype")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("only floating numeric intervals embed as quadratic pairs")
    valid = ((value.lower <= value.upper) & jnp.isfinite(value.lower)
             & jnp.isfinite(value.upper))
    return PairedQuadratic(
        constant_taylor_model(value.lower, source_size),
        constant_taylor_model(value.upper, source_size),
        valid,
    )


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
        return pair_from_interval(value, like.source_size)
    return constant_pair(value, like.source_size)


def _broadcast_pair(value: PairedQuadratic, shape) -> PairedQuadratic:
    return PairedQuadratic(
        _broadcast_model(value.lower, shape),
        _broadcast_model(value.upper, shape),
        jnp.broadcast_to(value.valid, shape),
    )


def _select(mask, yes: PairedQuadratic, no: PairedQuadratic) -> PairedQuadratic:
    return PairedQuadratic(
        _where_model(mask, yes.lower, no.lower),
        _where_model(mask, yes.upper, no.upper),
        jnp.where(mask, yes.valid, no.valid),
    )


def _add(x, y):
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
    """Interpretation ledger, signatures, and Interval producer provenance."""

    def __init__(self, *, capture_pairs=False):
        self.calls: list[tuple[str, str, Any]] = []
        self.signatures: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self.interval_events: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.failed_signatures: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self._interval_producers: dict[int, tuple[Interval, str]] = {}
        self.capture_pairs = capture_pairs
        self.pair_outputs: list[tuple[str, PairedQuadratic]] = []

    def add(self, primitive, mode, predicate=None):
        self.calls.append((primitive.name if hasattr(primitive, "name") else primitive, mode, predicate))

    @staticmethod
    def _kind(value):
        if isinstance(value, PairedQuadratic):
            return f"pair{value.shape}"
        if isinstance(value, Interval):
            if jnp.issubdtype(value.dtype, jnp.bool_):
                category = "Boolean control"
            elif jnp.issubdtype(value.dtype, jnp.integer):
                category = "discrete control"
            else:
                category = "numeric Interval"
            return f"{category}{value.shape}"
        if hasattr(value, "shape"):
            return f"point{value.shape}"
        return type(value).__name__

    def observe(self, primitive, args, answer):
        """Record abstract signatures and the creator of each Interval output."""
        name = primitive.name
        outputs = answer if isinstance(answer, (tuple, list)) else (answer,)
        self.signatures.append((name, tuple(self._kind(x) for x in args),
                                tuple(self._kind(x) for x in outputs)))
        upstream = tuple(
            self._interval_producers.get(id(x), (None, "external Interval"))[1]
            for x in args if isinstance(x, Interval)
        )
        for output in outputs:
            if self.capture_pairs and isinstance(output, PairedQuadratic):
                self.pair_outputs.append((name, output))
            if isinstance(output, Interval):
                origin = f"{name}({', '.join(self._kind(x) for x in args)})"
                self._interval_producers[id(output)] = (output, origin)
                self.interval_events.append((name, self._kind(output), origin, upstream))

    def observe_failure(self, primitive, args):
        upstream = tuple(
            self._interval_producers.get(id(x), (None, "external Interval"))[1]
            for x in args if isinstance(x, Interval)
        )
        self.failed_signatures.append((primitive.name,
                                       tuple(self._kind(x) for x in args), upstream))


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


def _product_polynomials(first: TaylorModel, second: TaylorModel):
    """Degree-2 product and interval overflow from Taylor's fixed-rank kernel."""
    return taylor_endpoint_models(first * second)


def _square_polynomial(model: TaylorModel):
    """Dependency-aware quadratic square, including interval square of q²."""
    return taylor_endpoint_models(_taylor_square(model))


def _native_input_valid(x: PairedQuadratic, y: PairedQuadratic | None = None):
    status = pair_status(x)
    valid = x.valid & status["zero_remainder"] & status["ordered"] & status["finite"]
    if y is not None:
        other = pair_status(y)
        valid = valid & y.valid & other["zero_remainder"] & other["ordered"] & other["finite"]
    return valid


def _finish_native(lower: TaylorModel, upper: TaylorModel, valid):
    # The analytic envelope proves pointwise order. The range-based shift
    # gives the representation its sufficient termwise order certificate.
    result = certify_pair_order(PairedQuadratic(lower, upper, valid))
    return _select(valid, result, PairedQuadratic(
        _nan_model_like(lower), _nan_model_like(upper), False,
    ))


def _reference_hull(lowers, uppers):
    """Constant reference shifts for a fixed small candidate arity."""
    lower_choices = []
    upper_choices = []
    for reference in lowers:  # Four product corners or two square endpoints.
        shift = jnp.max(jnp.stack([
            jnp.maximum(polynomial_range(reference - other).upper, 0.0)
            for other in lowers
        ]), axis=0)
        lower_choices.append(reference - shift)
    for reference in uppers:
        shift = jnp.max(jnp.stack([
            jnp.maximum(polynomial_range(other - reference).upper, 0.0)
            for other in uppers
        ]), axis=0)
        upper_choices.append(reference + shift)
    lower_scores = jnp.stack([polynomial_range(x).lower for x in lower_choices])
    upper_scores = jnp.stack([polynomial_range(x).upper for x in upper_choices])
    lower_index = jnp.argmax(lower_scores, axis=0)
    upper_index = jnp.argmin(upper_scores, axis=0)
    lower, upper = lower_choices[0], upper_choices[0]
    for index in range(1, len(lower_choices)):
        lower = _where_model(lower_index == index, lower_choices[index], lower)
        upper = _where_model(upper_index == index, upper_choices[index], upper)
    return lower, upper


def _four_candidate_product(x: PairedQuadratic, y: PairedQuadratic):
    products = (
        _product_polynomials(x.lower, y.lower),
        _product_polynomials(x.lower, y.upper),
        _product_polynomials(x.upper, y.lower),
        _product_polynomials(x.upper, y.upper),
    )
    general_lower, general_upper = _reference_hull(
        tuple(product[0] for product in products),
        tuple(product[1] for product in products),
    )
    x_positive = polynomial_range(x.lower).lower >= 0
    x_negative = polynomial_range(x.upper).upper <= 0
    y_positive = polynomial_range(y.lower).lower >= 0
    y_negative = polynomial_range(y.upper).upper <= 0
    lower, upper = general_lower, general_upper
    # The four sign cases select exact interval-product corners before the
    # degree-2 overflow enclosure. Branches have fixed arity independent of n.
    for gate, low_index, high_index in (
        (x_positive & y_positive, 0, 3),
        (x_negative & y_negative, 3, 0),
        (x_positive & y_negative, 2, 1),
        (x_negative & y_positive, 1, 2),
    ):
        lower = _where_model(gate, products[low_index][0], lower)
        upper = _where_model(gate, products[high_index][1], upper)
    fixed_sign = ((x_positive & y_positive) | (x_negative & y_negative)
                  | (x_positive & y_negative) | (x_negative & y_positive))
    _record(lax.mul_p, "native four-candidate fixed-sign", fixed_sign)
    return _finish_native(lower, upper, _native_input_valid(x, y))


def _midpoint_product(x: PairedQuadratic, y: PairedQuadratic):
    mx, dx = 0.5 * (x.lower + x.upper), 0.5 * (x.upper - x.lower)
    my, dy = 0.5 * (y.lower + y.upper), 0.5 * (y.upper - y.lower)
    rx, ry = polynomial_range(mx), polynomial_range(my)
    mux = jnp.maximum(jnp.abs(rx.lower), jnp.abs(rx.upper))
    muy = jnp.maximum(jnp.abs(ry.lower), jnp.abs(ry.upper))
    center_lower, center_upper = _product_polynomials(mx, my)
    _, width_upper = _product_polynomials(dx, dy)
    width = mux * dy + muy * dx + width_upper
    _record(lax.mul_p, "native midpoint", None)
    return _finish_native(center_lower - width, center_upper + width,
                          _native_input_valid(x, y))


def _square_pair(x: PairedQuadratic):
    lower_square = _square_polynomial(x.lower)
    upper_square = _square_polynomial(x.upper)
    positive = polynomial_range(x.lower).lower >= 0
    negative = polynomial_range(x.upper).upper <= 0
    _, crossing_upper = _reference_hull(
        (lower_square[0], upper_square[0]),
        (lower_square[1], upper_square[1]),
    )
    zero = constant_taylor_model(jnp.zeros_like(x.lower.constant), x.source_size)
    lower = _where_model(positive, lower_square[0],
                         _where_model(negative, upper_square[0], zero))
    upper = _where_model(positive, upper_square[1],
                         _where_model(negative, lower_square[1], crossing_upper))
    _record(lax.integer_pow_p, "native square", None)
    return _finish_native(lower, upper, _native_input_valid(x))


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


def _mul(x, y, *, multiplication_strategy="midpoint"):
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[lax.mul_p](x, y)
        return lax.mul_p.bind(x, y)
    if isinstance(x, PairedQuadratic) and not isinstance(y, (PairedQuadratic, Interval)):
        return _scale(x, y)
    if isinstance(y, PairedQuadratic) and not isinstance(x, (PairedQuadratic, Interval)):
        return _scale(y, x)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    if multiplication_strategy == "four_candidate":
        return _four_candidate_product(x, y)
    if multiplication_strategy == "midpoint":
        return _midpoint_product(x, y)
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


def _reciprocal_pair(y: PairedQuadratic) -> PairedQuadratic:
    """Tangent/secant reciprocal on certified fixed-sign endpoint ranges."""
    lower_range, upper_range = polynomial_range(y.lower), polynomial_range(y.upper)
    positive = lower_range.lower > 0
    negative = upper_range.upper < 0

    def tangent(model, bounds):
        center = 0.5 * bounds.lower + 0.5 * bounds.upper
        safe = jnp.where(center != 0, center, 1.0)
        return 2.0 / safe - model * (1.0 / (safe * safe))

    def secant(model, bounds):
        a = jnp.where(bounds.lower != 0, bounds.lower, 1.0)
        b = jnp.where(bounds.upper != 0, bounds.upper, 1.0)
        return (a + b - model) * (1.0 / (a * b))

    lower = _where_model(positive, tangent(y.upper, upper_range),
                         secant(y.upper, upper_range))
    upper = _where_model(positive, secant(y.lower, lower_range),
                         tangent(y.lower, lower_range))
    valid = _native_input_valid(y) & (positive | negative)
    _record("reciprocal.positive", "branch", positive)
    _record("reciprocal.negative", "branch", negative)
    record_pair_predicate("reciprocal.fixed_sign", positive | negative)
    return _finish_native(lower, upper, valid)


def _div_native(x, y, *, multiplication_strategy="midpoint"):
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[lax.div_p](x, y)
        return lax.div_p.bind(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    return _mul(x, _reciprocal_pair(y),
                multiplication_strategy=multiplication_strategy)


def _relu_polynomial(model: TaylorModel) -> tuple[TaylorModel, TaylorModel]:
    """Affine lower/chord upper bounds for ReLU of an exact quadratic."""
    bounds = polynomial_range(model)
    inactive = bounds.upper <= 0
    active = bounds.lower >= 0
    zero = constant_taylor_model(jnp.zeros_like(model.constant), model.source_size)
    span = bounds.upper - bounds.lower
    safe_span = jnp.where(span > 0, span, 1.0)
    chord = (model - bounds.lower) * (bounds.upper / safe_span)
    lower = _where_model(active, model, zero)
    upper = _where_model(active, model, _where_model(inactive, zero, chord))
    return lower, upper


def _max_polynomials(a: TaylorModel, b: TaylorModel):
    relu_lower, relu_upper = _relu_polynomial(a - b)
    return b + relu_lower, b + relu_upper


def _max_native(x, y):
    if not isinstance(x, PairedQuadratic) and not isinstance(y, PairedQuadratic):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return taylor_inclusion_registry[lax.max_p](x, y)
        return lax.max_p.bind(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    lower, _ = _max_polynomials(x.lower, y.lower)
    _, upper = _max_polynomials(x.upper, y.upper)
    return _finish_native(lower, upper, _native_input_valid(x, y))


def _min_native(x, y):
    return _neg(_max_native(_neg(x), _neg(y)))


def _abs_native(x):
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[lax.abs_p](x)
        return lax.abs_p.bind(x)
    positive = polynomial_range(x.lower).lower >= 0
    negative = polynomial_range(x.upper).upper <= 0
    zero = constant_taylor_model(jnp.zeros_like(x.lower.constant), x.source_size)
    first_lower, _ = _max_polynomials(x.lower, zero)
    crossing_lower, _ = _max_polynomials(first_lower, -x.upper)
    _, crossing_upper = _max_polynomials(-x.lower, x.upper)
    lower = _where_model(positive, x.lower,
                         _where_model(negative, -x.upper, crossing_lower))
    upper = _where_model(positive, x.upper,
                         _where_model(negative, -x.lower, crossing_upper))
    _record("abs.fixed_positive", "branch", positive)
    _record("abs.fixed_negative", "branch", negative)
    return _finish_native(lower, upper, _native_input_valid(x))


def _sqrt_native(x, **_):
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[lax.sqrt_p](x)
        return lax.sqrt_p.bind(x)
    lower_range, upper_range = polynomial_range(x.lower), polynomial_range(x.upper)
    domain = lower_range.lower >= 0
    a, b = jnp.maximum(lower_range.lower, 0.0), jnp.maximum(lower_range.upper, 0.0)
    span = b - a
    slope = (jnp.sqrt(b) - jnp.sqrt(a)) / jnp.where(span > 0, span, 1.0)
    lower = jnp.sqrt(a) + (x.lower - a) * slope
    center = jnp.maximum(0.5 * upper_range.lower + 0.5 * upper_range.upper, 0.0)
    safe_center = jnp.where(center > 0, center, 1.0)
    root_center = jnp.sqrt(safe_center)
    tangent = root_center + (x.upper - safe_center) * (0.5 / root_center)
    zero = constant_taylor_model(jnp.zeros_like(x.upper.constant), x.source_size)
    upper = _where_model(upper_range.upper == 0, zero, tangent)
    record_pair_predicate("sqrt.nonnegative", domain)
    return _finish_native(lower, upper, _native_input_valid(x) & domain)


def _curvature_unary_polynomial(model: TaylorModel, kind: str,
                                curvature_bound=1.0):
    """Degree-2 envelopes from a certified absolute curvature bound."""
    bounds = polynomial_range(model)
    center = 0.5 * bounds.lower + 0.5 * bounds.upper
    delta = model - center
    _, square_upper = _square_polynomial(delta)
    if kind == "sin":
        core = jnp.sin(center) + delta * jnp.cos(center)
    else:
        core = jnp.cos(center) - delta * jnp.sin(center)
    return (core - 0.5 * curvature_bound * square_upper,
            core + 0.5 * curvature_bound * square_upper)


def _trig_native(kind: str, x, *, curvature="global"):
    primitive = lax.sin_p if kind == "sin" else lax.cos_p
    if not isinstance(x, PairedQuadratic):
        if isinstance(x, Interval):
            return taylor_inclusion_registry[primitive](x)
        return primitive.bind(x)
    global_range = pair_range(x)
    midpoint = 0.5 * global_range.lower + 0.5 * global_range.upper
    critical_offset = jnp.pi / 2 if kind == "sin" else 0.0
    # Absence of an interior critical point makes the function monotone on
    # the entire certified scalar range. The conservative equality test also
    # routes ranges touching a critical point through the constant hull.
    low_sector = jnp.floor((global_range.lower - critical_offset) / jnp.pi)
    high_sector = jnp.floor((global_range.upper - critical_offset) / jnp.pi)
    monotone = low_sector == high_sector
    if curvature == "local":
        f_lower = (jnp.sin(global_range.lower) if kind == "sin"
                   else jnp.cos(global_range.lower))
        f_upper = (jnp.sin(global_range.upper) if kind == "sin"
                   else jnp.cos(global_range.upper))
        # On a certified monotone sector, |f| reaches its maximum at one
        # interval endpoint. Since f''=-f, this bounds Taylor curvature.
        curvature_bound = jnp.maximum(jnp.abs(f_lower), jnp.abs(f_upper))
    elif curvature == "global":
        curvature_bound = 1.0
    else:
        raise ValueError(f"unknown native trig curvature bound: {curvature}")
    increasing = (jnp.cos(midpoint) >= 0 if kind == "sin"
                  else -jnp.sin(midpoint) >= 0)
    source_lower = _where_model(increasing, x.lower, x.upper)
    source_upper = _where_model(increasing, x.upper, x.lower)
    lower, _ = _curvature_unary_polynomial(
        source_lower, kind, curvature_bound)
    _, upper = _curvature_unary_polynomial(
        source_upper, kind, curvature_bound)
    constant_lower = constant_taylor_model(-jnp.ones_like(x.lower.constant), x.source_size)
    constant_upper = constant_taylor_model(jnp.ones_like(x.upper.constant), x.source_size)
    lower = _where_model(monotone, lower, constant_lower)
    upper = _where_model(monotone, upper, constant_upper)
    _record(f"{kind}.monotone", "branch", monotone)
    _record(f"{kind}.critical_constant_hull", "branch", ~monotone)
    return _finish_native(lower, upper, _native_input_valid(x))


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


def _binary_mask_pair(value: Interval, like: PairedQuadratic) -> PairedQuadratic:
    """Relax a discrete 0/1 arithmetic mask only at its dot-product use.

    The selector remains a discrete Interval. For the bilinear summand
    ``x_i * mask_i``, the extrema over {0, 1} and [0, 1] coincide, so a
    constant endpoint pair is a sound local product operand. This does not
    promote integer candidate indices in other operations.
    """
    valid = ((value.lower >= 0) & (value.upper <= 1)
             & (value.lower <= value.upper))
    record_pair_predicate("dot_general.binary_mask", valid)
    numeric = Interval(value.lower.astype(like.dtype),
                       value.upper.astype(like.dtype))
    converted = pair_from_interval(numeric, like.source_size)
    return PairedQuadratic(converted.lower, converted.upper,
                           converted.valid & valid)


def _dot_general(a, b, *, multiplication_strategy="midpoint", **params):
    if not isinstance(a, PairedQuadratic) and not isinstance(b, PairedQuadratic):
        return lax.dot_general_p.bind(a, b, **params)
    like = _template(a, b)
    if isinstance(a, Interval) and not jnp.issubdtype(a.dtype, jnp.floating):
        a = _binary_mask_pair(a, like)
    if isinstance(b, Interval) and not jnp.issubdtype(b.dtype, jnp.floating):
        b = _binary_mask_pair(b, like)
    a, b = _promote(a, like), _promote(b, like)
    (ac, bc), (ab, bb) = params["dimension_numbers"]
    a = _moveaxis(a, ab + ac, tuple(range(len(ab) + len(ac))))
    b = _moveaxis(b, bb + bc, tuple(range(len(bb) + len(bc))))
    batch_n, contract_n = len(ab), len(ac)
    a_free = a.shape[batch_n + contract_n:]
    b_free = b.shape[batch_n + contract_n:]
    a = a.reshape(a.shape + (1,) * len(b_free))
    b = b.reshape(b.shape[:batch_n + contract_n] + (1,) * len(a_free) + b_free)
    products = _mul(a, b, multiplication_strategy=multiplication_strategy)
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
    if not any(isinstance(x, PairedQuadratic) for x in cases):
        if isinstance(which, Interval) or any(isinstance(x, Interval) for x in cases):
            # Jaxpr literals can be TypedFloat values without ``.shape``;
            # normalize them before the existing interval selector inspects
            # case shapes. This remains a discrete/Boolean Interval result.
            normalized = tuple(
                Interval(jnp.asarray(x.lower), jnp.asarray(x.upper))
                if isinstance(x, Interval) else jnp.asarray(x)
                for x in cases
            )
            return taylor_inclusion_registry[lax.select_n_p](which, *normalized)
        return lax.select_n_p.bind(which, *cases)
    like = _template(*cases)
    values = [_promote(x, like) for x in cases]
    if isinstance(which, Interval):
        if not (jnp.issubdtype(which.dtype, jnp.bool_)
                or jnp.issubdtype(which.dtype, jnp.integer)):
            raise TypeError("paired selection requires a Boolean or integer control interval")
        shape = jnp.broadcast_shapes(which.shape, *(value.shape for value in values))
        values = [_broadcast_pair(value, shape) for value in values]
        low, high = jnp.broadcast_to(which.lower, shape), jnp.broadcast_to(which.upper, shape)
        selector_valid = ((low <= high) & (low >= 0) & (high < len(values)))
        feasible = [(low <= index) & (high >= index)
                    for index in range(len(values))]
        # The first feasible complete branch is a sound reference at each
        # output component. Arity is fixed by the traced select primitive.
        reference = values[0]
        found = jnp.zeros(shape, dtype=bool)
        all_valid = selector_valid
        for mask, value in zip(feasible, values):
            reference = _select(mask & ~found, value, reference)
            found = found | mask
            status = pair_status(value)
            branch_valid = (value.valid & status["zero_remainder"]
                            & status["ordered"] & status["finite"])
            all_valid = all_valid & jnp.where(mask, branch_valid, True)
        masked_lower = tuple(_where_model(mask, value.lower, reference.lower)
                             for mask, value in zip(feasible, values))
        masked_upper = tuple(_where_model(mask, value.upper, reference.upper)
                             for mask, value in zip(feasible, values))
        hull_lower, hull_upper = _reference_hull(masked_lower, masked_upper)
        selected_lower = taylor_inclusion_registry[lax.select_n_p](
            low, *(value.lower for value in values))
        selected_upper = taylor_inclusion_registry[lax.select_n_p](
            low, *(value.upper for value in values))
        fixed = low == high
        lower = _where_model(fixed, selected_lower, hull_lower)
        upper = _where_model(fixed, selected_upper, hull_upper)
        record_pair_predicate("select_n.selector_valid", selector_valid)
        record_pair_predicate("select_n.feasible_branches_valid", all_valid)
        return _finish_native(lower, upper, all_valid & found)
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
    lax.integer_pow_p,
    lax.concatenate_p, lax.select_n_p, lax.scatter_p, lax.scatter_add_p,
}
_fallback_primitives: set[Any] = set()
for _primitive in (lax.sin_p, lax.cos_p, lax.max_p, lax.min_p, lax.abs_p, lax.sqrt_p):
    register_pair_taylor_fallback(_primitive)


def register_pair_rule(primitive, rule: Callable[..., Any]):
    """Register a primitive owner rule in the opt-in paired interpreter."""
    pair_inclusion_registry[primitive] = rule


pair_native_inclusion_registry: dict[Any, Callable[..., Any]] = {}
PAIR_NATIVE_FAMILIES = (
    "mixed", "division", "nonsmooth", "sqrt", "trigonometry", "jacobian",
)


def register_pair_native_rule(primitive, rule: Callable[..., Any]):
    """Register an owner rule used by the explicit native operator path."""
    pair_native_inclusion_registry[primitive] = rule


def _pair_jit_rule(*args, multiplication_strategy="midpoint",
                   operator_strategy="legacy", native_families=None,
                   trig_curvature="global", **params):
    closed = params.pop("jaxpr")
    if isinstance(closed, jax.extend.core.ClosedJaxpr):
        return pair_jaxpr(closed.jaxpr, closed.consts, *args,
                          multiplication_strategy=multiplication_strategy,
                          operator_strategy=operator_strategy,
                          native_families=native_families,
                          trig_curvature=trig_curvature)
    return pair_jaxpr(closed, [], *args,
                      multiplication_strategy=multiplication_strategy,
                      operator_strategy=operator_strategy,
                      native_families=native_families,
                      trig_curvature=trig_curvature)


def _integer_square(x, *, y, multiplication_strategy):
    if y != 2:
        raise NotImplementedError("paired integer_pow only supports square (y=2)")
    if multiplication_strategy == "taylor" or not isinstance(x, PairedQuadratic):
        return _fallback(lax.integer_pow_p, x, y=y)
    return _square_pair(x)

try:
    pair_inclusion_registry[jax._src.pjit.jit_p] = _pair_jit_rule
except AttributeError:  # pragma: no cover
    pass


def _is_abstract(value):
    return isinstance(value, (PairedQuadratic, Interval))


def pair_jaxpr(jaxpr, consts, *args, multiplication_strategy="midpoint",
               operator_strategy="legacy", native_families=None,
               trig_curvature="global"):
    audit = _active_audit.get()
    families = (frozenset(PAIR_NATIVE_FAMILIES) if operator_strategy == "native"
                else frozenset()) if native_families is None else frozenset(native_families)
    unknown = families.difference(PAIR_NATIVE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown paired native rule families: {sorted(unknown)}")
    registry = dict(pair_inclusion_registry)
    native_registered = set()
    if "jacobian" in families:
        registry.update(pair_native_inclusion_registry)
        native_registered.update(pair_native_inclusion_registry)
    if "mixed" not in families:
        def legacy_mixed(primitive, native):
            def rule(*values, **params):
                if (any(isinstance(x, PairedQuadratic) for x in values)
                        and any(isinstance(x, Interval) for x in values)):
                    return _fallback(primitive, *values, **params)
                return native(*values, **params)
            return rule
        registry[lax.add_p] = legacy_mixed(lax.add_p, _add)
        registry[lax.select_n_p] = lambda which, *cases: (
            _fallback(lax.select_n_p, which, *cases)
            if (isinstance(which, Interval) and any(isinstance(x, PairedQuadratic) for x in cases))
            or (any(isinstance(x, PairedQuadratic) for x in cases)
                and any(isinstance(x, Interval) for x in cases))
            else _select_n(which, *cases)
        )
    else:
        registry[lax.add_p] = _add
        registry[lax.select_n_p] = _select_n
    if "division" in families:
        registry[lax.div_p] = lambda x, y: _div_native(
            x, y, multiplication_strategy=multiplication_strategy,
        )
    if "nonsmooth" in families:
        registry[lax.abs_p] = _abs_native
        registry[lax.max_p] = _max_native
        registry[lax.min_p] = _min_native
        native_registered.update((lax.abs_p, lax.max_p, lax.min_p))
    if "sqrt" in families:
        registry[lax.sqrt_p] = _sqrt_native
        native_registered.add(lax.sqrt_p)
    if "trigonometry" in families:
        registry[lax.sin_p] = lambda x, **_: _trig_native(
            "sin", x, curvature=trig_curvature)
        registry[lax.cos_p] = lambda x, **_: _trig_native(
            "cos", x, curvature=trig_curvature)
        native_registered.update((lax.sin_p, lax.cos_p))
    registry[lax.mul_p] = lambda x, y: _mul(
        x, y, multiplication_strategy=multiplication_strategy,
    ) if "mixed" in families or not (
        (isinstance(x, PairedQuadratic) or isinstance(y, PairedQuadratic))
        and (isinstance(x, Interval) or isinstance(y, Interval))
    ) else _fallback(lax.mul_p, x, y)
    registry[lax.dot_general_p] = lambda x, y, **p: _dot_general(
        x, y, multiplication_strategy=multiplication_strategy, **p,
    ) if "mixed" in families or not (
        (isinstance(x, PairedQuadratic) or isinstance(y, PairedQuadratic))
        and (isinstance(x, Interval) or isinstance(y, Interval))
    ) else _fallback(lax.dot_general_p, x, y, **p)
    registry[lax.integer_pow_p] = lambda x, *, y: _integer_square(
        x, y=y, multiplication_strategy=multiplication_strategy,
    )
    try:
        registry[jax._src.pjit.jit_p] = lambda *values, **p: _pair_jit_rule(
            *values, multiplication_strategy=multiplication_strategy,
            operator_strategy=operator_strategy, native_families=tuple(families),
            trig_curvature=trig_curvature, **p,
        )
    except AttributeError:  # pragma: no cover
        pass
    if audit is not None:
        def audited(primitive, rule):
            def apply(*values, **params):
                try:
                    answer = rule(*values, **params)
                except Exception:
                    audit.observe_failure(primitive, values)
                    raise
                audit.observe(primitive, values, answer)
                return answer
            return apply
        registry = {primitive: audited(primitive, rule)
                    for primitive, rule in registry.items()}
    return interpret_inclusion_jaxpr(
        jaxpr, consts, *args,
        registry=registry,
        is_abstract=_is_abstract,
        label="paired quadratic",
        on_rule=(lambda primitive: audit.add(
            primitive,
            "Taylor fallback" if primitive in _fallback_primitives
            and primitive not in native_registered else "native",
        )) if audit is not None else None,
    )


def pqif(function: Callable[..., Any], *, multiplication_strategy="midpoint",
         operator_strategy="legacy", native_families=None,
         trig_curvature="global") -> Callable[..., Any]:
    """Transform a point function through ordered quadratic endpoint pairs.

    ``multiplication_strategy`` is fixed before tracing. The default uses the
    pair-native midpoint/half-width rule. The original pair-to-Taylor rule and
    the tighter four-candidate rule remain explicit alternatives.
    ``operator_strategy='legacy'`` preserves the explicit nonproduct Taylor
    fallbacks; ``'native'`` selects the experimental pair-native live rules.
    ``native_families`` is a static diagnostic override of individual rule
    families, shared with nested JIT interpretation.
    """

    if multiplication_strategy not in ("taylor", "four_candidate", "midpoint"):
        raise ValueError(f"unknown paired multiplication strategy: {multiplication_strategy}")
    if operator_strategy not in ("legacy", "native"):
        raise ValueError(f"unknown paired operator strategy: {operator_strategy}")
    if trig_curvature not in ("global", "local"):
        raise ValueError(f"unknown native trig curvature bound: {trig_curvature}")
    if native_families is not None:
        native_families = tuple(native_families)
        unknown = set(native_families).difference(PAIR_NATIVE_FAMILIES)
        if unknown:
            raise ValueError(f"unknown paired native rule families: {sorted(unknown)}")

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
            outputs = pair_jaxpr(
                closed.jaxpr, closed.literals, *flat_args,
                multiplication_strategy=multiplication_strategy,
                operator_strategy=operator_strategy,
                native_families=native_families,
                trig_curvature=trig_curvature,
            )
        finally:
            _active_audit.reset(token)
        return outputs[0] if len(outputs) == 1 else outputs

    return wrapped
