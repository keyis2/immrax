"""Paired quadratic enclosures: ordered lower/upper quadratic endpoints.

A PairedQuadratic represents [L(xi), U(xi)] on one shared normalized source
box, with L and U stored as remainder-free QuadraticPolynomial endpoints.
Besides exact affine operations it provides the midpoint product, square,
fixed-sign reciprocal and division, sqrt, sin/cos, abs, max/min and selection
rules; omitted higher-degree terms are enclosed in the endpoint coefficients.
There is no Taylor remainder, conversion or fallback: every rule returns two
degree-two polynomials directly. Internal rules own their pointwise endpoint
order; validity is elementwise and certificate callers require all(valid).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .interval import Interval
from .quadratic_polynomial import (
    QuadraticPolynomial,
    constant_quadratic,
    evaluate_quadratic,
    normalized_quadratic_seed,
    quadratic_product,
    quadratic_range,
    quadratic_square,
)


@register_pytree_node_class
class PairedQuadratic:
    """Two degree-2 endpoint polynomials over shared normalized sources.

    Direct construction is trusted and low level: the caller owns the
    pointwise order L(xi) <= U(xi) on the source box and records any failure
    to establish it in ``valid``. Only static shape and dtype are checked.
    """

    __array_ufunc__ = None

    def __init__(self, lower: QuadraticPolynomial, upper: QuadraticPolynomial, valid=True):
        if not isinstance(lower, QuadraticPolynomial) or not isinstance(upper, QuadraticPolynomial):
            raise TypeError("paired endpoints must be QuadraticPolynomial values")
        if lower.shape != upper.shape or lower.source_size != upper.source_size:
            raise ValueError("paired endpoints must have equal value/source shapes")
        if lower.dtype != upper.dtype:
            raise ValueError("paired endpoints must have the same floating dtype")
        self.lower = lower
        self.upper = upper
        self.valid = jnp.broadcast_to(jnp.asarray(valid), lower.shape)

    def tree_flatten(self):
        return ((self.lower, self.upper, self.valid), None)

    @classmethod
    def tree_unflatten(cls, _, children):
        value = object.__new__(cls)
        value.lower, value.upper, value.valid = children
        return value

    @property
    def shape(self):
        return self.lower.shape

    @property
    def ndim(self):
        return self.lower.ndim

    @property
    def size(self):
        return self.lower.size

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
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return PairedQuadratic(self.lower.reshape(*shape), self.upper.reshape(*shape),
                               self.valid.reshape(*shape))

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return PairedQuadratic(self.lower.transpose(*axes), self.upper.transpose(*axes),
                               jnp.transpose(self.valid, axes))

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(self, other)

    def __neg__(self):
        return PairedQuadratic(-self.upper, -self.lower, self.valid)

    def __sub__(self, other):
        return _add(self, -other if isinstance(other, PairedQuadratic) else -_point(other))

    def __rsub__(self, other):
        return _add(-self, other)

    def __mul__(self, other):
        return pair_multiply(self, other) if isinstance(other, PairedQuadratic) else _scale(self, other)

    def __rmul__(self, other):
        return _scale(self, other)

    def __truediv__(self, other):
        return pair_divide(self, other)

    def __rtruediv__(self, other):
        return pair_divide(other, self)


def _point(value):
    if isinstance(value, (PairedQuadratic, QuadraticPolynomial, Interval)):
        raise TypeError("exact pair operation requires a plain point operand")
    return jnp.asarray(value)


def _add(x: PairedQuadratic, y) -> PairedQuadratic:
    if isinstance(y, PairedQuadratic):
        lower, upper = x.lower + y.lower, x.upper + y.upper
        return PairedQuadratic(lower, upper, jnp.broadcast_to(x.valid & y.valid, lower.shape))
    y = _point(y)
    lower, upper = x.lower + y, x.upper + y
    return PairedQuadratic(lower, upper, jnp.broadcast_to(x.valid, lower.shape))


def _where(mask, yes: QuadraticPolynomial, no: QuadraticPolynomial) -> QuadraticPolynomial:
    return QuadraticPolynomial(
        jnp.where(mask, yes.constant, no.constant),
        jnp.where(mask[..., None], yes.linear, no.linear),
        jnp.where(mask[..., None, None], yes.quadratic, no.quadratic),
    )


def _scale(x: PairedQuadratic, scale) -> PairedQuadratic:
    """Scale both endpoints; a negative factor swaps their roles."""

    scale = _point(scale)
    lower, upper = x.lower * scale, x.upper * scale
    nonnegative = jnp.broadcast_to(scale >= 0, lower.shape)
    valid = jnp.broadcast_to(x.valid & jnp.isfinite(scale), lower.shape)
    return PairedQuadratic(_where(nonnegative, lower, upper), _where(nonnegative, upper, lower), valid)


def evaluate_pair(value: PairedQuadratic, sources) -> tuple[jax.Array, jax.Array]:
    """Evaluate both endpoints at one normalized source point."""

    return evaluate_quadratic(value.lower, sources), evaluate_quadratic(value.upper, sources)


def pair_range(value: PairedQuadratic) -> Interval:
    """Termwise lower range of L and upper range of U on the unit source box."""

    return Interval(quadratic_range(value.lower).lower, quadratic_range(value.upper).upper)


def pair_status(value: PairedQuadratic) -> dict[str, jax.Array]:
    """Elementwise diagnostic status: validity, finite coefficients and termwise order.

    ``ordered`` is the sufficient termwise test ``range(U - L).lower >= 0``.
    It is not necessary: an analytically ordered pair may fail it, so it does
    not refute the order an internal rule established.
    """
    margin = quadratic_range(value.upper - value.lower).lower
    return {"valid": value.valid, "finite": _finite(value.lower) & _finite(value.upper) & jnp.isfinite(margin),
            "ordering_margin": margin, "ordered": margin >= 0}


def constant_pair(value, source_size) -> PairedQuadratic:
    polynomial = constant_quadratic(value, source_size)
    return PairedQuadratic(polynomial, polynomial)


def pair_from_interval(value: Interval, source_size) -> PairedQuadratic:
    """Embed one floating Interval as constant endpoints, valid where ordered and finite.

    Boolean and integer Intervals are control values, not continuous pairs.
    """
    if not isinstance(value, Interval) or not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("only floating numeric Intervals embed as quadratic pairs")
    valid = (value.lower <= value.upper) & jnp.isfinite(value.lower) & jnp.isfinite(value.upper)
    return PairedQuadratic(constant_quadratic(value.lower, source_size),
                           constant_quadratic(value.upper, source_size), valid)


def normalized_pair_seed(lower, upper) -> PairedQuadratic:
    polynomial = normalized_quadratic_seed(lower, upper)
    return PairedQuadratic(polynomial, polynomial)


def _finite(x: QuadraticPolynomial):
    return (jnp.isfinite(x.constant) & jnp.all(jnp.isfinite(x.linear), axis=-1)
            & jnp.all(jnp.isfinite(x.quadratic), axis=(-2, -1)))


def _input_valid(*values: PairedQuadratic):
    # Per output element: ``valid`` and finite coefficients and order margin of
    # the same element, so a bad element never poisons the others.
    # Pointwise order is the input's own invariant and is not rechecked.
    valid = True
    for x in values:
        margin = quadratic_range(x.upper - x.lower).lower
        valid = valid & x.valid & _finite(x.lower) & _finite(x.upper) & jnp.isfinite(margin)
    return valid


def _finish(lower: QuadraticPolynomial, upper: QuadraticPolynomial, valid) -> PairedQuadratic:
    # The analytic envelope proves pointwise order from ordered inputs; the
    # endpoints are returned unshifted. Failed premises poison the output.
    valid = jnp.broadcast_to(valid, lower.shape)
    return PairedQuadratic(_where(valid, lower, _nan_like(lower)),
                           _where(valid, upper, _nan_like(upper)), valid)


def _nan_like(x: QuadraticPolynomial) -> QuadraticPolynomial:
    return QuadraticPolynomial(jnp.full_like(x.constant, jnp.nan), jnp.full_like(x.linear, jnp.nan),
                               jnp.full_like(x.quadratic, jnp.nan))


def _select(mask, yes: PairedQuadratic, no: PairedQuadratic) -> PairedQuadratic:
    return PairedQuadratic(_where(mask, yes.lower, no.lower), _where(mask, yes.upper, no.upper),
                           jnp.where(mask, yes.valid, no.valid))


def _magnitude(x: QuadraticPolynomial):
    bounds = quadratic_range(x)
    return jnp.maximum(jnp.abs(bounds.lower), jnp.abs(bounds.upper))


def pair_multiply(x: PairedQuadratic, y: PairedQuadratic) -> PairedQuadratic:
    """Native midpoint/half-width product of two ordered pairs.

    With M = (L+U)/2 and D = (U-L)/2 >= 0 pointwise, every concrete product is
    Mx My + Mx ey + My ex + ex ey with |ex| <= Dx and |ey| <= Dy. Hence it lies
    within Mx My -/+ (mu_x Dy + mu_y Dx + Dx Dy), mu = max |range(M)|. The two
    degree-2 products are enclosed by ``quadratic_product``; only the upper
    enclosure of Dx Dy >= 0 is used, so the width is pointwise nonnegative.
    """

    mx, dx = 0.5 * (x.lower + x.upper), 0.5 * (x.upper - x.lower)
    my, dy = 0.5 * (y.lower + y.upper), 0.5 * (y.upper - y.lower)
    center_lower, center_upper = quadratic_product(mx, my)
    _, width_upper = quadratic_product(dx, dy)
    width = _magnitude(mx) * dy + _magnitude(my) * dx + width_upper
    return _finish(center_lower - width, center_upper + width, _input_valid(x, y))


def _choose(index, choices):
    chosen = choices[0]
    for i in range(1, len(choices)):
        chosen = _where(index == i, choices[i], chosen)
    return chosen


def _lower_reference_hull(lowers):
    """Greatest termwise-range lower polynomial below every candidate.

    Each candidate is shifted down by a constant until it lies termwise below
    all others; a fixed small arity (branches or square endpoints) is used.
    """

    choices = []
    for reference in lowers:
        shift = jnp.max(jnp.stack([
            jnp.maximum(quadratic_range(reference - other).upper, 0.0) for other in lowers
        ]), axis=0)
        choices.append(reference - shift)
    return _choose(jnp.argmax(jnp.stack([quadratic_range(c).lower for c in choices]), axis=0), choices)


def _upper_reference_hull(uppers):
    """Least termwise-range upper polynomial above every candidate."""

    choices = []
    for reference in uppers:
        shift = jnp.max(jnp.stack([
            jnp.maximum(quadratic_range(other - reference).upper, 0.0) for other in uppers
        ]), axis=0)
        choices.append(reference + shift)
    return _choose(jnp.argmin(jnp.stack([quadratic_range(c).upper for c in choices]), axis=0), choices)


def pair_square(x: PairedQuadratic) -> PairedQuadratic:
    """Square: [L^2, U^2] if L >= 0, [U^2, L^2] if U <= 0, else [0, max]."""

    lower_square, upper_square = quadratic_square(x.lower), quadratic_square(x.upper)
    positive = quadratic_range(x.lower).lower >= 0
    negative = quadratic_range(x.upper).upper <= 0
    crossing_upper = _upper_reference_hull((lower_square[1], upper_square[1]))
    zero = constant_quadratic(jnp.zeros_like(x.lower.constant), x.source_size)
    lower = _where(positive, lower_square[0], _where(negative, upper_square[0], zero))
    upper = _where(positive, upper_square[1], _where(negative, lower_square[1], crossing_upper))
    return _finish(lower, upper, _input_valid(x))


def pair_reciprocal(y: PairedQuadratic) -> PairedQuadratic:
    """Tangent/secant reciprocal on certified fixed-sign endpoint ranges.

    On a sign-definite interval 1/t is monotone decreasing and either convex
    (t > 0) or concave (t < 0). For t > 0 the tangent at the midpoint of
    range(U) lies below 1/U and the chord over range(L) lies above 1/L; t < 0
    swaps tangent and chord. A zero-crossing range is invalid.
    """

    lower_range, upper_range = quadratic_range(y.lower), quadratic_range(y.upper)
    positive = lower_range.lower > 0
    negative = upper_range.upper < 0

    def tangent(p, bounds):
        center = 0.5 * bounds.lower + 0.5 * bounds.upper
        safe = jnp.where(center != 0, center, 1.0)
        return 2.0 / safe - p * (1.0 / (safe * safe))

    def secant(p, bounds):
        a = jnp.where(bounds.lower != 0, bounds.lower, 1.0)
        b = jnp.where(bounds.upper != 0, bounds.upper, 1.0)
        return (a + b - p) * (1.0 / (a * b))

    lower = _where(positive, tangent(y.upper, upper_range), secant(y.upper, upper_range))
    upper = _where(positive, secant(y.lower, lower_range), tangent(y.lower, lower_range))
    return _finish(lower, upper, _input_valid(y) & (positive | negative))


def _exact_singleton(value: PairedQuadratic):
    lower, upper = value.lower, value.upper
    return ((lower.constant == upper.constant)
            & jnp.all(lower.linear == upper.linear, axis=-1)
            & jnp.all(lower.quadratic == upper.quadratic, axis=(-2, -1)))


def pair_divide(x, y) -> PairedQuadratic:
    """x / y as the midpoint product with ``pair_reciprocal(y)``.

    A point denominator, or an exact constant singleton pair, is applied by one
    reciprocal scaling so that a singleton numerator stays a singleton.
    """

    if not isinstance(y, PairedQuadratic):
        return _scale(x, jnp.reciprocal(_point(y)))
    if not isinstance(x, PairedQuadratic):
        x = constant_pair(_point(x), y.source_size)
    general = pair_multiply(x, pair_reciprocal(y))
    c = y.lower.constant
    point = (_exact_singleton(y) & y.valid
             & jnp.all(y.lower.linear == 0, axis=-1)
             & jnp.all(y.lower.quadratic == 0, axis=(-2, -1))
             & jnp.isfinite(c) & (c != 0))
    scaled = _scale(x, jnp.reciprocal(jnp.where(point, c, 1.0)))
    return _select(jnp.broadcast_to(point, general.shape), scaled, general)


def pair_integer_pow(x: PairedQuadratic, exponent: int) -> PairedQuadratic:
    if exponent == 0:
        one = constant_quadratic(jnp.ones_like(x.lower.constant), x.source_size)
        return PairedQuadratic(one, one, x.valid)
    if exponent == 1:
        return x
    if exponent == 2:
        return pair_square(x)
    raise NotImplementedError("paired integer power supports exponents 0, 1, and 2")


# ------------------------------------------------- unary and nonsmooth rules
# Every rule owns its pointwise
# order analytically and returns per-element validity through ``_finish``.

def _zero_like(x: QuadraticPolynomial) -> QuadraticPolynomial:
    return constant_quadratic(jnp.zeros_like(x.constant), x.source_size)


def _promote_pair(value, like: PairedQuadratic) -> PairedQuadratic:
    if isinstance(value, PairedQuadratic):
        if value.source_size != like.source_size:
            raise ValueError("paired operands have different source dimensions")
        return value
    return constant_pair(_point(value).astype(like.dtype), like.source_size)


def _relu_polynomial(p: QuadraticPolynomial):
    """Affine lower and chord upper bounds for ReLU of one exact quadratic."""
    bounds = quadratic_range(p)
    inactive, active = bounds.upper <= 0, bounds.lower >= 0
    zero = _zero_like(p)
    span = bounds.upper - bounds.lower
    chord = (p - bounds.lower) * (bounds.upper / jnp.where(span > 0, span, 1.0))
    return _where(active, p, zero), _where(active, p, _where(inactive, zero, chord))


def _max_polynomials(a: QuadraticPolynomial, b: QuadraticPolynomial):
    relu_lower, relu_upper = _relu_polynomial(a - b)
    return b + relu_lower, b + relu_upper


def pair_max(x, y) -> PairedQuadratic:
    """max via b + ReLU(a - b) on each endpoint; a plain -inf operand is neutral."""
    neutral, pair = None, None
    if isinstance(x, PairedQuadratic) != isinstance(y, PairedQuadratic):
        plain_first = not isinstance(x, PairedQuadratic)
        plain, pair = (x, y) if plain_first else (y, x)
        plain = _point(plain)
        neutral = jnp.broadcast_to(jnp.isneginf(plain), jnp.broadcast_shapes(plain.shape, pair.shape))
        plain = jnp.where(jnp.isneginf(plain), jnp.zeros_like(plain), plain)
        x, y = (plain, pair) if plain_first else (pair, plain)
    like = x if isinstance(x, PairedQuadratic) else y
    x, y = _promote_pair(x, like), _promote_pair(y, like)
    lower, _ = _max_polynomials(x.lower, y.lower)
    _, upper = _max_polynomials(x.upper, y.upper)
    result = _finish(lower, upper, _input_valid(x, y))
    if neutral is None:
        return result
    shape = neutral.shape
    pair = PairedQuadratic(pair.lower + jnp.zeros(shape, pair.dtype), pair.upper + jnp.zeros(shape, pair.dtype),
                           jnp.broadcast_to(pair.valid, shape))
    return PairedQuadratic(_where(neutral, pair.lower, result.lower), _where(neutral, pair.upper, result.upper),
                           jnp.where(neutral, pair.valid, result.valid))


def pair_min(x, y) -> PairedQuadratic:
    negate = lambda v: -v if isinstance(v, PairedQuadratic) else -_point(v)
    return -pair_max(negate(x), negate(y))


def pair_abs(x: PairedQuadratic) -> PairedQuadratic:
    positive = quadratic_range(x.lower).lower >= 0
    negative = quadratic_range(x.upper).upper <= 0
    first_lower, _ = _max_polynomials(x.lower, _zero_like(x.lower))
    crossing_lower, _ = _max_polynomials(first_lower, -x.upper)
    _, crossing_upper = _max_polynomials(-x.lower, x.upper)
    lower = _where(positive, x.lower, _where(negative, -x.upper, crossing_lower))
    upper = _where(positive, x.upper, _where(negative, -x.lower, crossing_upper))
    return _finish(lower, upper, _input_valid(x))


def pair_sqrt(x: PairedQuadratic) -> PairedQuadratic:
    """Concave sqrt: chord over range(L) below, tangent at mid range(U) above."""
    lower_range, upper_range = quadratic_range(x.lower), quadratic_range(x.upper)
    domain = lower_range.lower >= 0
    a, b = jnp.maximum(lower_range.lower, 0.0), jnp.maximum(lower_range.upper, 0.0)
    span = b - a
    slope = (jnp.sqrt(b) - jnp.sqrt(a)) / jnp.where(span > 0, span, 1.0)
    lower = jnp.sqrt(a) + (x.lower - a) * slope
    center = jnp.maximum(0.5 * upper_range.lower + 0.5 * upper_range.upper, 0.0)
    safe_center = jnp.where(center > 0, center, 1.0)
    root_center = jnp.sqrt(safe_center)
    tangent = root_center + (x.upper - safe_center) * (0.5 / root_center)
    upper = _where(upper_range.upper == 0, _zero_like(x.upper), tangent)
    return _finish(lower, upper, _input_valid(x) & domain)


def _trig_range(kind, lower, upper):
    """Range from interval endpoints and every possible interior extremum."""
    function = jnp.sin if kind == "sin" else jnp.cos
    lo = jnp.minimum(function(lower), function(upper))
    hi = jnp.maximum(function(lower), function(upper))
    max_offset = jnp.pi / 2 if kind == "sin" else 0.0
    min_offset = -jnp.pi / 2 if kind == "sin" else jnp.pi
    period = 2 * jnp.pi

    def contains(offset):
        return jnp.ceil((lower-offset)/period) <= jnp.floor((upper-offset)/period)

    broad = upper-lower >= period
    return (jnp.where(broad | contains(min_offset), -1.0, lo),
            jnp.where(broad | contains(max_offset), 1.0, hi))


def _curvature_unary(p: QuadraticPolynomial, kind: str):
    """Signed local-curvature Taylor envelopes of sin/cos(p).

    The range of p contains the segment from its range midpoint to p(xi).
    On that interval f''=-f, so Taylor's integral remainder lies between
    k_lower*delta**2/2 and k_upper*delta**2/2. Degree reduction of delta**2
    selects the opposite square endpoint when its curvature is negative.
    """
    bounds = quadratic_range(p)
    center = 0.5 * bounds.lower + 0.5 * bounds.upper
    delta = p - center
    square_lower, square_upper = quadratic_square(delta)
    f_lower, f_upper = _trig_range(kind, bounds.lower, bounds.upper)
    curvature_lower, curvature_upper = -f_upper, -f_lower
    if kind == "sin":
        core = jnp.sin(center) + delta * jnp.cos(center)
    else:
        core = jnp.cos(center) - delta * jnp.sin(center)
    lower_square = _where(curvature_lower >= 0, square_lower, square_upper)
    upper_square = _where(curvature_upper >= 0, square_upper, square_lower)
    return (core + lower_square * (0.5 * curvature_lower),
            core + upper_square * (0.5 * curvature_upper))


def pair_trig(kind: str, x: PairedQuadratic) -> PairedQuadratic:
    """Signed local-curvature sin/cos rule with the singleton critical branch.

    On a monotone sector the increasing (decreasing) endpoint feeds each side.
    A range touching a critical point uses the signed-curvature envelope of
    an exact singleton's one polynomial, and otherwise the constant [-1, 1].
    """
    global_range = pair_range(x)
    midpoint = 0.5 * global_range.lower + 0.5 * global_range.upper
    critical_offset = jnp.pi / 2 if kind == "sin" else 0.0
    low_sector = jnp.floor((global_range.lower - critical_offset) / jnp.pi)
    high_sector = jnp.floor((global_range.upper - critical_offset) / jnp.pi)
    monotone = low_sector == high_sector
    increasing = jnp.cos(midpoint) >= 0 if kind == "sin" else -jnp.sin(midpoint) >= 0
    lower, _ = _curvature_unary(_where(increasing, x.lower, x.upper), kind)
    _, upper = _curvature_unary(_where(increasing, x.upper, x.lower), kind)
    singleton_lower, singleton_upper = _curvature_unary(x.lower, kind)
    singleton = _exact_singleton(x)
    minus_one = constant_quadratic(-jnp.ones_like(x.lower.constant), x.source_size)
    one = constant_quadratic(jnp.ones_like(x.upper.constant), x.source_size)
    lower = _where(monotone, lower, _where(singleton, singleton_lower, minus_one))
    upper = _where(monotone, upper, _where(singleton, singleton_upper, one))
    return _finish(lower, upper, _input_valid(x))


def pair_select_n(low, high, cases) -> PairedQuadratic:
    """Select among paired cases under a control interval [low, high].

    A fixed selector (low == high) selects exactly. Otherwise every feasible
    branch is masked into a constant-shift reference hull, so the result
    encloses whichever branch the concrete selector takes. Validity requires
    a well-formed selector and valid feasible branches only.
    """
    like = next(case for case in cases if isinstance(case, PairedQuadratic))
    values = [_promote_pair(case, like) for case in cases]
    shape = jnp.broadcast_shapes(jnp.shape(low), jnp.shape(high), *(v.shape for v in values))
    values = [PairedQuadratic(v.lower + jnp.zeros(shape, v.dtype), v.upper + jnp.zeros(shape, v.dtype),
                              jnp.broadcast_to(v.valid, shape)) for v in values]
    low = jnp.broadcast_to(jnp.asarray(low).astype(jnp.int32), shape)
    high = jnp.broadcast_to(jnp.asarray(high).astype(jnp.int32), shape)
    selector_valid = (low <= high) & (low >= 0) & (high < len(values))
    feasible = [(low <= i) & (high >= i) for i in range(len(values))]
    reference, found, branches_valid = values[0], jnp.zeros(shape, bool), selector_valid
    for mask, value in zip(feasible, values):
        first = mask & ~found
        reference = PairedQuadratic(_where(first, value.lower, reference.lower),
                                    _where(first, value.upper, reference.upper), reference.valid)
        found = found | mask
        branches_valid = branches_valid & jnp.where(mask, _input_valid(value), True)
    hull_lower = _lower_reference_hull([_where(m, v.lower, reference.lower) for m, v in zip(feasible, values)])
    hull_upper = _upper_reference_hull([_where(m, v.upper, reference.upper) for m, v in zip(feasible, values)])
    index = jnp.clip(low, 0, len(values) - 1)
    selected_lower = _choose(index, [v.lower for v in values])
    selected_upper = _choose(index, [v.upper for v in values])
    fixed = low == high
    return _finish(_where(fixed, selected_lower, hull_lower), _where(fixed, selected_upper, hull_upper),
                   branches_valid & found)
