"""Degree-2 polynomials over the normalized source box.

A QuadraticPolynomial represents p(xi) = c + l^T xi + xi^T Q xi / 2 for
xi in [-1, 1]^m. It carries no interval remainder. Operator arithmetic is
exact; ``quadratic_product`` and ``quadratic_square`` return lower/upper
degree-2 polynomials whose constants absorb the omitted degree-3/4 range.
Source identity is not tracked; operands must share one normalized seed.
"""

from __future__ import annotations

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


def quadratic_product(x: QuadraticPolynomial, y: QuadraticPolynomial):
    """Lower and upper degree-2 polynomials enclosing x(xi) y(xi) pointwise.

    With x = c1 + l1 + q1 and y = c2 + l2 + q2 (q the half-weighted quadratic
    form), the kept part is c1 c2 + (c1 l2 + c2 l1) + (c1 q2 + c2 q1 + l1 l2).
    The omitted l1 q2 + q1 l2 + q1 q2 is bounded termwise and added to the
    lower/upper constants. This is the zero-remainder Taylor product kernel.
    """

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
    omitted = Interval(first.lower + (second.lower + third.lower),
                       first.upper + (second.upper + third.upper))
    return _with_constant_bounds(c, a, q, omitted)


def quadratic_square(x: QuadraticPolynomial):
    """Dependency-aware degree-2 enclosure of x(xi)^2.

    The omitted part 2 l q + q^2 uses the interval square of q's range rather
    than an independent product of that range with itself.
    """

    c = x.constant**2
    a = 2 * x.linear * x.constant[..., None]
    q = (
        2 * x.quadratic * x.constant[..., None, None]
        + 2 * x.linear[..., :, None] * x.linear[..., None, :]
    )
    linear, quadratic = _component_ranges(x)
    cross, square = _imul(linear, quadratic), _isquare(quadratic)
    omitted = Interval(2 * cross.lower + square.lower, 2 * cross.upper + square.upper)
    return _with_constant_bounds(c, a, q, omitted)
