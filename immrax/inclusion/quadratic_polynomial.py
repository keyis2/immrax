"""Degree-2 polynomials over the normalized source box.

A QuadraticPolynomial represents p(xi) = c + l^T xi + xi^T Q xi / 2 for
xi in [-1, 1]^m. It carries no interval remainder. Operator arithmetic is
exact; ``quadratic_product`` and ``quadratic_square`` return lower/upper
degree-2 polynomials whose constants absorb the omitted degree-3/4 range.
Source identity is not tracked; operands must share one normalized seed.
"""

from __future__ import annotations

import itertools
import operator

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .interval import Interval


@register_pytree_node_class
class QuadraticPolynomial:
    """Dense coefficients of shapes ``s``, ``s + (m,)`` and ``s + (m, m)``."""

    # A NumPy left operand would otherwise broadcast this object elementwise
    # into an object array instead of deferring to the reflected operator.
    __array_ufunc__ = None

    def __init__(self, constant, linear, quadratic):
        constant, linear, quadratic = (jnp.asarray(constant), jnp.asarray(linear),
                                       jnp.asarray(quadratic))
        if linear.ndim != constant.ndim + 1 or linear.shape[:-1] != constant.shape:
            raise ValueError(
                "linear shape must be value shape + (source_size,); got "
                f"{linear.shape} for value shape {constant.shape}"
            )
        m = linear.shape[-1]
        if quadratic.shape != constant.shape + (m, m):
            raise ValueError(
                "quadratic shape must be value shape + (source_size, source_size); "
                f"got {quadratic.shape}"
            )
        if not (constant.dtype == linear.dtype == quadratic.dtype):
            raise ValueError("polynomial coefficients must share one dtype")
        if not jnp.issubdtype(constant.dtype, jnp.floating):
            raise TypeError("polynomial coefficients must have a floating dtype")
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

    def tree_flatten(self):
        return ((self.constant, self.linear, self.quadratic), None)

    @classmethod
    def tree_unflatten(cls, _, children):
        # JAX may unflatten with non-array leaves (tracers, batching specs);
        # validation belongs to user construction, not to tree reconstruction.
        value = object.__new__(cls)
        value.constant, value.linear, value.quadratic = children
        return value

    @property
    def shape(self):
        return self.constant.shape

    @property
    def ndim(self):
        return self.constant.ndim

    @property
    def size(self):
        return self.constant.size

    @property
    def source_size(self):
        return self.linear.shape[-1]

    @property
    def dtype(self):
        return self.constant.dtype

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        return QuadraticPolynomial(self.constant[index], self.linear[index], self.quadratic[index])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        m = self.source_size
        return QuadraticPolynomial(
            self.constant.reshape(*shape),
            self.linear.reshape(*shape, m),
            self.quadratic.reshape(*shape, m, m),
        )

    def transpose(self, *axes):
        n = self.ndim
        if not axes:
            axes = tuple(reversed(range(n)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return QuadraticPolynomial(
            jnp.transpose(self.constant, axes),
            jnp.transpose(self.linear, axes + (n,)),
            jnp.transpose(self.quadratic, axes + (n, n + 1)),
        )

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(self, other)

    def __neg__(self):
        return QuadraticPolynomial(-self.constant, -self.linear, -self.quadratic)

    def __sub__(self, other):
        return _add(self, -other if isinstance(other, QuadraticPolynomial) else -_point(other))

    def __rsub__(self, other):
        return _add(-self, other)

    def __mul__(self, other):
        return _scale(self, other)

    def __rmul__(self, other):
        return _scale(self, other)

    def __truediv__(self, other):
        return _scale(self, other, divide=True)


def _point(value):
    if isinstance(value, (QuadraticPolynomial, Interval)):
        raise TypeError("exact polynomial operation requires a plain point operand")
    return jnp.asarray(value)


def _broadcast(x: QuadraticPolynomial, shape) -> QuadraticPolynomial:
    m = x.source_size
    return QuadraticPolynomial(
        jnp.broadcast_to(x.constant, shape),
        jnp.broadcast_to(x.linear, shape + (m,)),
        jnp.broadcast_to(x.quadratic, shape + (m, m)),
    )


def _add(x: QuadraticPolynomial, y) -> QuadraticPolynomial:
    if isinstance(y, QuadraticPolynomial):
        if y.source_size != x.source_size:
            raise ValueError("polynomials have different source dimensions")
        shape = jnp.broadcast_shapes(x.shape, y.shape)
        x, y = _broadcast(x, shape), _broadcast(y, shape)
        return QuadraticPolynomial(x.constant + y.constant, x.linear + y.linear,
                                   x.quadratic + y.quadratic)
    y = _point(y)
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    x = _broadcast(x, shape)
    constant = x.constant + y
    dtype = constant.dtype
    return QuadraticPolynomial(constant, x.linear.astype(dtype), x.quadratic.astype(dtype))


def _scale(x: QuadraticPolynomial, scale, *, divide=False) -> QuadraticPolynomial:
    """Exact coefficientwise scaling; endpoint order is the pair layer's concern."""

    scale = _point(scale)
    shape = jnp.broadcast_shapes(x.shape, scale.shape)
    x = _broadcast(x, shape)
    scale = jnp.broadcast_to(scale, shape)
    apply = operator.truediv if divide else operator.mul
    return QuadraticPolynomial(
        apply(x.constant, scale),
        apply(x.linear, scale[..., None]),
        apply(x.quadratic, scale[..., None, None]),
    )


def constant_quadratic(value, source_size) -> QuadraticPolynomial:
    """Lift a point value into a normalized source space of static size."""

    value = jnp.asarray(value)
    m = operator.index(source_size)
    if m < 0:
        raise ValueError("source size must be nonnegative")
    dtype = jnp.result_type(value, float)
    value = value.astype(dtype)
    return QuadraticPolynomial(
        value,
        jnp.zeros(value.shape + (m,), dtype=dtype),
        jnp.zeros(value.shape + (m, m), dtype=dtype),
    )


def normalized_quadratic_seed(lower, upper) -> QuadraticPolynomial:
    """Affine map from [-1, 1]^m onto one physical box; poison invalid boxes."""

    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    if lower.shape != upper.shape:
        raise ValueError("seed endpoints must have equal shapes")
    if lower.ndim != 1 or lower.size == 0:
        raise ValueError("the seed requires one nonempty vector input")
    valid = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper)) & jnp.all(lower <= upper)
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    m = center.size
    dtype = jnp.result_type(center, float)
    poison = jnp.asarray(jnp.nan, dtype=dtype)
    return QuadraticPolynomial(
        jnp.where(valid, center.astype(dtype), poison),
        jnp.where(valid, jnp.diag(radius.astype(dtype)), poison),
        jnp.zeros((m, m, m), dtype=dtype),
    )


def evaluate_quadratic(x: QuadraticPolynomial, sources) -> jax.Array:
    """Evaluate at one normalized source point of shape ``(source_size,)``."""

    sources = jnp.asarray(sources)
    return (
        x.constant
        + jnp.einsum("...i,i->...", x.linear, sources)
        + 0.5 * jnp.einsum("...ij,i,j->...", x.quadratic, sources, sources)
    )


def _component_ranges(x: QuadraticPolynomial) -> tuple[Interval, Interval]:
    """Termwise ranges of l^T xi and xi^T Q xi / 2 on [-1, 1]^m.

    Bounds |l|_1 for the linear term, the signed diagonal Q_ii/2 terms
    separately, and |sym(Q)_ij| for each off-diagonal pair i < j.
    """

    linear_radius = jnp.sum(jnp.abs(x.linear), axis=-1)
    diagonal = 0.5 * jnp.diagonal(x.quadratic, axis1=-2, axis2=-1)
    diagonal_lower = jnp.sum(jnp.minimum(diagonal, 0), axis=-1)
    diagonal_upper = jnp.sum(jnp.maximum(diagonal, 0), axis=-1)
    symmetric = 0.5 * (x.quadratic + jnp.swapaxes(x.quadratic, -1, -2))
    upper_mask = jnp.triu(jnp.ones((x.source_size, x.source_size), dtype=bool), k=1)
    off_diagonal_radius = jnp.sum(jnp.where(upper_mask, jnp.abs(symmetric), 0), axis=(-2, -1))
    return (Interval(-linear_radius, linear_radius),
            Interval(diagonal_lower - off_diagonal_radius, diagonal_upper + off_diagonal_radius))


def quadratic_range(x: QuadraticPolynomial) -> Interval:
    """Termwise range on [-1, 1]^m."""

    linear, quadratic = _component_ranges(x)
    return Interval(x.constant + (linear.lower + quadratic.lower),
                    x.constant + (linear.upper + quadratic.upper))


def _imul(x: Interval, y: Interval) -> Interval:
    products = jnp.stack((x.lower * y.lower, x.lower * y.upper,
                          x.upper * y.lower, x.upper * y.upper))
    return Interval(jnp.min(products, axis=0), jnp.max(products, axis=0))


def _isquare(x: Interval) -> Interval:
    contains_zero = (x.lower <= 0) & (x.upper >= 0)
    return Interval(jnp.where(contains_zero, 0, jnp.minimum(x.lower**2, x.upper**2)),
                    jnp.maximum(x.lower**2, x.upper**2))


def _with_constant_bounds(c, a, q, omitted: Interval):
    """Close a degree-2 truncation by moving its omitted range into constants."""

    return (QuadraticPolynomial(c + omitted.lower, a, q),
            QuadraticPolynomial(c + omitted.upper, a, q))


def _canonical_indices(m, degree):
    # Static nondecreasing index tuples; empty arrays when there are no sources.
    rows = list(itertools.combinations_with_replacement(range(m), degree))
    return tuple(jnp.asarray(v, dtype=int) for v in
                 (zip(*rows) if rows else [()] * degree))


def cubic_monomial_indices(source_size):
    """Static canonical ``i <= j <= k`` indices for :func:`reduce_cubic_monomials`.

    All degree-three monomials occur once, in lexicographic order. For zero
    sources, return three empty integer arrays. Callers form and collect their
    own coefficients in this order before applying the shared reduction.
    """
    m = operator.index(source_size)
    if m < 0:
        raise ValueError("source size must be nonnegative")
    return _canonical_indices(m, 3)


def _cubic_product_coefficients(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Canonical cubic monomials, with signed like terms collected first."""
    # Source indices depend only on static shape; numerical work stays in JAX.
    i, j, k = cubic_monomial_indices(x.source_size)
    sx = 0.5 * (x.quadratic + jnp.swapaxes(x.quadratic, -1, -2))
    sy = 0.5 * (y.quadratic + jnp.swapaxes(y.quadratic, -1, -2))

    def term(a, b, c):
        weight = jnp.where(b == c, 0.5, 1.0)
        return weight * (
            x.linear[..., a] * sy[..., b, c] + y.linear[..., a] * sx[..., b, c])

    coefficients = (term(i, j, k)
                    + jnp.where(i < j, term(j, i, k), 0)
                    + jnp.where(j < k, term(k, i, j), 0))
    return i, j, k, coefficients


def _collected_cubic_radius(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Absolute-coefficient radius, with O(m^3/6) canonical coefficients."""
    if x.source_size == 0:
        return jnp.zeros_like(x.constant)
    return jnp.sum(jnp.abs(_cubic_product_coefficients(x, y)[-1]), axis=-1)


def _chebyshev_cubic(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Chebyshev cubic reduction of one product."""
    if x.source_size == 0:
        return jnp.zeros_like(x.linear), jnp.zeros_like(x.constant)
    return reduce_cubic_monomials(*_cubic_product_coefficients(x, y), x.source_size)


def reduce_cubic_monomials(i, j, k, coefficients, source_size):
    """Return a Chebyshev linear shift and cubic residual radius on [-1,1]^m.

    ``coefficients[..., r]`` multiplies ``z[i[r]] z[j[r]] z[k[r]]`` directly,
    with no factorial or half-weighting. The one-dimensional integer index
    arrays must describe distinct canonical monomials ``0 <= i <= j <= k < m``;
    omitted monomials have zero coefficient. Collect all signed contributions
    to each monomial across the entire operation before calling this kernel.
    ``source_size`` is the static nonnegative source dimension ``m``.

    The returned arrays have shapes ``coefficients.shape[:-1] + (m,)`` and
    ``coefficients.shape[:-1]``. They retain .75*t from t^3 and .5*s from
    t^2*s, t != s. Leading value axes, jit, vmap, and m=0 are supported.

    The residuals are .25*T3(t), .5*T2(t)*s, or an all-distinct monomial,
    where Tn is the nth Chebyshev polynomial. Since every factor is bounded
    by one, multiplicity weights .25/.5/1 give an absolute-coefficient radius
    enclosing the cubic minus the returned linear form on the entire unit box.
    """
    same = i == k
    repeated = (i == j) | (j == k)
    target = jnp.where(i == j, k, i)
    retained = jnp.where(same, .75, jnp.where(repeated, .5, 0.)) * coefficients
    linear = jnp.zeros(coefficients.shape[:-1] + (source_size,), coefficients.dtype)
    linear = linear.at[..., target].add(retained)
    radius = jnp.sum(jnp.abs(coefficients) * jnp.where(same, .25, jnp.where(repeated, .5, 1.)), axis=-1)
    return linear, radius


def _chebyshev_quartic(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Full quartic Chebyshev reduction of one product; see :func:`_reduce_quartic`."""
    if x.source_size == 0:
        return jnp.zeros_like(x.constant), jnp.zeros_like(x.quadratic), jnp.zeros_like(x.constant)
    return _reduce_quartic(*_quartic_product_coefficients(x, y), x.source_size)


def _quartic_product_coefficients(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Canonical i<=j<=k<=l quartic monomials of q1 q2: C(m+3,4), never m^4."""
    i, j, k, l = _canonical_indices(x.source_size, 4)
    sx = .5 * (x.quadratic + jnp.swapaxes(x.quadratic, -1, -2))
    sy = .5 * (y.quadratic + jnp.swapaxes(y.quadratic, -1, -2))

    def partition(a, b, c, d):
        # Canonical quadratic monomial coefficients: Q_ii/2 or sym(Q)_ij.
        weight = jnp.where(a == b, .5, 1.) * jnp.where(c == d, .5, 1.)
        forward = sx[..., a, b] * sy[..., c, d]
        reverse = sy[..., a, b] * sx[..., c, d]
        return weight * (forward + jnp.where((a == c) & (b == d), 0., reverse))

    # Three unordered pair partitions, omitting repeated partitions of a multiset.
    coefficients = (partition(i, j, k, l)
                    + jnp.where(j < k, partition(i, k, j, l), 0.)
                    + jnp.where((i < j) & (k < l), partition(i, l, j, k), 0.))
    return i, j, k, l, coefficients


def _reduce_quartic(i, j, k, l, coefficients, m):
    """Unit-box Chebyshev reduction of canonical quartic coefficients.

    For distinct letters retain t^2-1/8 from t^4, .75*t*s from t^3*s,
    .5*(t^2+s^2)-.25 from t^2*s^2, and .5*s*u from t^2*s*u.
    The respective residuals have radii 1/8, 1/4, 1/4, 1/2; four-distinct
    terms have radius 1. Scatters return the half-weighted Q convention.
    """
    same = i == l
    triple = ((i == k) & (k < l)) | ((i < j) & (j == l))
    two_pairs = (i == j) & (j < k) & (k == l)
    left_pair = (i == j) & (j < k) & (k < l)
    middle_pair = (i < j) & (j == k) & (k < l)
    right_pair = (i < j) & (j < k) & (k == l)
    one_pair = left_pair | middle_pair | right_pair
    constant = -jnp.sum(coefficients * jnp.where(same, .125, jnp.where(two_pairs, .25, 0.)), axis=-1)
    q = jnp.zeros(coefficients.shape[:-1] + (m, m), coefficients.dtype)
    q = q.at[..., i, i].add(coefficients * jnp.where(same, 2., jnp.where(two_pairs, 1., 0.)))
    q = q.at[..., k, k].add(jnp.where(two_pairs, coefficients, 0.))
    row, col = jnp.where(left_pair, k, i), jnp.where(right_pair, j, l)
    off = coefficients * jnp.where(triple, .75, jnp.where(one_pair, .5, 0.))
    q = q.at[..., row, col].add(off).at[..., col, row].add(off)
    weight = jnp.where(same, .125, jnp.where(triple | two_pairs, .25, jnp.where(one_pair, .5, 1.)))
    radius = jnp.sum(jnp.abs(coefficients) * weight, axis=-1)
    return constant, q, radius


def quadratic_product(x: QuadraticPolynomial, y: QuadraticPolynomial, *, collect_cubic=False,
                      reduction="auto"):
    """Lower and upper degree-2 polynomials enclosing x(xi) y(xi) pointwise.

    With x = c1 + l1 + q1 and y = c2 + l2 + q2 (q the half-weighted quadratic
    form), the kept part is c1 c2 + (c1 l2 + c2 l1) + (c1 q2 + c2 q1 + l1 l2).
    By default cubic terms use Chebyshev reduction (``"chebyshev3"``).
    Explicit ``reduction=None`` uses component-range products for the omitted
    l1 q2 + q1 l2 + q1 q2. With ``collect_cubic=True``, equal cubic monomials
    are collected before bounding their signed coefficients. Intersecting with
    the component cubic bound prevents enlargement. The quartic q1 q2 keeps its
    structured interval product; no fourth-order tensor is formed. These ranges
    shift constants.

    Collection and Chebyshev3 cost O(m^3) work/storage versus legacy O(m^2), with
    m(m+1)(m+2)/6 canonical cubic coefficients. The option must be static under
    JIT (close over it or use ``static_argnames=("collect_cubic",)``).

    Alternatively static ``reduction="chebyshev3"`` retains lower-degree
    Chebyshev cubic terms, or ``"chebyshev4"`` also reduces all quartics.
    These change the retained polynomial and use their own analytic remainder,
    without intersecting ranges around different centers. Full quartic costs
    O(m^4) canonical work/storage. ``"auto"`` selects Chebyshev3 unless
    ``collect_cubic=True`` selects collection; explicit conflicting policies fail.
    """

    if reduction == "auto":
        reduction = None if collect_cubic else "chebyshev3"
    if reduction not in (None, "chebyshev3", "chebyshev4"):
        raise ValueError("reduction must be None, 'chebyshev3', or 'chebyshev4'")
    if collect_cubic and reduction is not None:
        raise ValueError("choose collect_cubic or a Chebyshev reduction policy")
    if x.source_size != y.source_size:
        raise ValueError("polynomials have different source dimensions")
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    x, y = _broadcast(x, shape), _broadcast(y, shape)
    c = x.constant * y.constant
    a = x.linear * y.constant[..., None] + y.linear * x.constant[..., None]
    q = (
        x.quadratic * y.constant[..., None, None]
        + y.quadratic * x.constant[..., None, None]
        + x.linear[..., :, None] * y.linear[..., None, :]
        + y.linear[..., :, None] * x.linear[..., None, :]
    )
    lx, qx = _component_ranges(x)
    ly, qy = _component_ranges(y)
    first, second, third = _imul(lx, qy), _imul(qx, ly), _imul(qx, qy)
    if reduction is not None:
        linear, cubic_radius = _chebyshev_cubic(x, y)
        a = a + linear
        if reduction == "chebyshev4":
            constant, quadratic, quartic_radius = _chebyshev_quartic(x, y)
            c, q = c + constant, q + quadratic
            radius = cubic_radius + quartic_radius
            omitted = Interval(-radius, radius)
        else:
            omitted = Interval(third.lower - cubic_radius, third.upper + cubic_radius)
    elif collect_cubic:
        cubic_radius = _collected_cubic_radius(x, y)
        omitted = Interval(jnp.maximum(first.lower + second.lower, -cubic_radius) + third.lower,
                           jnp.minimum(first.upper + second.upper, cubic_radius) + third.upper)
    else:
        omitted = Interval(first.lower + (second.lower + third.lower),
                           first.upper + (second.upper + third.upper))
    return _with_constant_bounds(c, a, q, omitted)


def _value_axis(axis, ndim):
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"product axis {axis} is out of range for value rank {ndim}")
    return axis % ndim


def quadratic_sum_products(x: QuadraticPolynomial, y: QuadraticPolynomial, *, axis=0,
                           collect_cubic=False, reduction="auto"):
    """Lower and upper degree-2 polynomials enclosing sum_axis x y pointwise.

    ``axis`` is a value axis of the broadcast operands, never a source axis.
    Retained degree<=2 parts add exactly. Signed canonical cubic (and, for
    ``"chebyshev4"``, quartic) monomial coefficients are summed over the axis
    before any absolute value or Chebyshev remainder, so cross-product
    cancellation is kept. The default policy is fused Chebyshev3; explicit
    ``reduction=None`` equals the sum of legacy :func:`quadratic_product`
    endpoints. With ``collect_cubic`` the sum of
    separate collected cubic bounds is intersected with the fused radius,
    valid because both enclose the same omitted polynomial. Unreduced quartics
    q1 q2 keep the per-product structured interval bound. Options are static
    and follow :func:`quadratic_product`.
    """

    if reduction == "auto":
        reduction = None if collect_cubic else "chebyshev3"
    if reduction not in (None, "chebyshev3", "chebyshev4"):
        raise ValueError("reduction must be None, 'chebyshev3', or 'chebyshev4'")
    if collect_cubic and reduction is not None:
        raise ValueError("choose collect_cubic or a Chebyshev reduction policy")
    if x.source_size != y.source_size:
        raise ValueError("polynomials have different source dimensions")
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    axis = _value_axis(axis, len(shape))
    x, y = _broadcast(x, shape), _broadcast(y, shape)
    m = x.source_size
    c = jnp.sum(x.constant * y.constant, axis=axis)
    a = jnp.sum(x.linear * y.constant[..., None] + y.linear * x.constant[..., None], axis=axis)
    q = jnp.sum(x.quadratic * y.constant[..., None, None] + y.quadratic * x.constant[..., None, None]
                + x.linear[..., :, None] * y.linear[..., None, :]
                + y.linear[..., :, None] * x.linear[..., None, :], axis=axis)
    lx, qx = _component_ranges(x)
    ly, qy = _component_ranges(y)
    first, second, third = _imul(lx, qy), _imul(qx, ly), _imul(qx, qy)
    total = lambda value: jnp.sum(value, axis=axis)
    if reduction is not None or collect_cubic:
        *cubic_index, cubic = _cubic_product_coefficients(x, y)
        separate_radius = jnp.sum(jnp.abs(cubic), axis=-1)
        cubic = total(cubic)
    if reduction is not None:
        linear, cubic_radius = reduce_cubic_monomials(*cubic_index, cubic, m)
        a = a + linear
        if reduction == "chebyshev4":
            *quartic_index, quartic = _quartic_product_coefficients(x, y)
            constant, quadratic, quartic_radius = _reduce_quartic(*quartic_index, total(quartic), m)
            c, q = c + constant, q + quadratic
            radius = cubic_radius + quartic_radius
            omitted = Interval(-radius, radius)
        else:
            omitted = Interval(total(third.lower) - cubic_radius, total(third.upper) + cubic_radius)
    elif collect_cubic:
        # Separate collected bounds, then the fused signed radius: all
        # enclose the same omitted cubic about the same retained centre.
        cubic_radius = jnp.sum(jnp.abs(cubic), axis=-1)
        separate = Interval(total(jnp.maximum(first.lower + second.lower, -separate_radius)),
                            total(jnp.minimum(first.upper + second.upper, separate_radius)))
        omitted = Interval(jnp.maximum(separate.lower, -cubic_radius) + total(third.lower),
                           jnp.minimum(separate.upper, cubic_radius) + total(third.upper))
    else:
        omitted = Interval(total(first.lower + (second.lower + third.lower)),
                           total(first.upper + (second.upper + third.upper)))
    return _with_constant_bounds(c, a, q, omitted)


def quadratic_square(x: QuadraticPolynomial, *, reduction="chebyshev3"):
    """Dependency-aware degree-2 enclosure of x(xi)^2.

    Cubic 2 l q terms use Chebyshev3 by default; explicit ``reduction=None``
    keeps the legacy component range. The quartic q^2 always uses the interval
    square of q's range rather than an independent product with itself.
    """

    if reduction not in (None, "chebyshev3"):
        raise ValueError("square reduction must be None or 'chebyshev3'")
    c = x.constant**2
    a = 2 * x.linear * x.constant[..., None]
    q = (
        2 * x.quadratic * x.constant[..., None, None]
        + 2 * x.linear[..., :, None] * x.linear[..., None, :]
    )
    linear, quadratic = _component_ranges(x)
    square = _isquare(quadratic)
    if reduction == "chebyshev3":
        retained, radius = _chebyshev_cubic(x, x)
        a = a + retained
        omitted = Interval(square.lower - radius, square.upper + radius)
    else:
        cross = _imul(linear, quadratic)
        omitted = Interval(2 * cross.lower + square.lower, 2 * cross.upper + square.upper)
    return _with_constant_bounds(c, a, q, omitted)
