"""Experimental degree-2 Taylor models over one normalized source box.

This module is intentionally small and incomplete.  It supports the primitive
surface needed by the fixed-layout MuJoDiCo hinge experiment, not a general
Taylor-model transform.  A model represents

    c + a @ xi + 0.5 * xi.T @ Q @ xi + R,  xi in [-1, 1]**n.

For every fixed source ``xi``, the contract is that the represented value lies
in the fiber

    c + a @ xi + 0.5 * xi.T @ Q @ xi + R.

This pointwise contract is stronger than containment in the model's global
range and is what permits sound downstream composition.  ``Q`` uses ordinary
dense storage; only its symmetric part affects the represented polynomial.
``R`` is an :class:`Interval` shared over the source box.  Polynomial ranges
are computed term by term, so correlations between distinct monomials are not
used when forming a global interval.

Multiplication keeps terms through degree two and encloses the discarded
cubic and quartic terms in ``R``.  Smooth unary functions use a second-order
expansion about ``c`` and an interval Lagrange remainder.  Arithmetic uses
ordinary JAX floating point and has not received a directed-rounding audit.

Current deliberate limitations include a fixed normalized box, dense ``Q``,
termwise polynomial ranges, interval-only fallbacks at singular-but-defined
smooth domains, top enclosures outside supported real domains, one flat
transformed output, and an incomplete primitive surface.  Crossing ReLUs keep
the existing sources through a linear relaxation; arbitrary uncertain
selection uses an interval hull.  Custom primitive rules live in a mutable
experimental registry and may be registered by the modules that own those
primitives.  All nonconstant Taylor operands in one expression
must descend from the same normalized seed.  The representation records only
the common source dimension: source-space identity and merging independently
seeded source spaces are unsupported.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import operator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import ad_util
from jax._src.core import Literal
from jax.tree_util import register_pytree_node_class

from .affine import AffineBound
from .interval import Interval


def _iadd(x: Interval, y: Interval) -> Interval:
    return Interval(x.lower + y.lower, x.upper + y.upper)


def _ineg(x: Interval) -> Interval:
    return Interval(lax.neg_p.bind(x.upper), lax.neg_p.bind(x.lower))


def _isub(x: Interval, y: Interval) -> Interval:
    return _iadd(x, _ineg(y))


def _imul(x: Interval, y: Interval) -> Interval:
    products = jnp.stack(
        (
            x.lower * y.lower,
            x.lower * y.upper,
            x.upper * y.lower,
            x.upper * y.upper,
        )
    )
    return Interval(jnp.min(products, axis=0), jnp.max(products, axis=0))


def _iscale(x: Interval, scale) -> Interval:
    scale = jnp.asarray(scale)
    return Interval(
        jnp.where(scale >= 0, scale * x.lower, scale * x.upper),
        jnp.where(scale >= 0, scale * x.upper, scale * x.lower),
    )


def _isquare(x: Interval) -> Interval:
    contains_zero = (x.lower <= 0) & (x.upper >= 0)
    return Interval(
        jnp.where(contains_zero, 0, jnp.minimum(x.lower**2, x.upper**2)),
        jnp.maximum(x.lower**2, x.upper**2),
    )


def _ipow3(x: Interval) -> Interval:
    return Interval(x.lower**3, x.upper**3)


def _isum(x: Interval, axis=None) -> Interval:
    return Interval(jnp.sum(x.lower, axis=axis), jnp.sum(x.upper, axis=axis))


def _zero_interval(value) -> Interval:
    value = jnp.asarray(value)
    zero = jnp.zeros_like(value)
    return Interval(zero, zero)


def _as_interval(value) -> Interval:
    if isinstance(value, Interval):
        return value
    value = jnp.asarray(value)
    return Interval(value, value)


def _ireciprocal(x: Interval) -> Interval:
    excludes_zero = (x.lower > 0) | (x.upper < 0)
    safe_lower = jnp.where(excludes_zero, x.lower, 1)
    safe_upper = jnp.where(excludes_zero, x.upper, 1)
    first, second = 1 / safe_lower, 1 / safe_upper
    infinity = jnp.full_like(first, jnp.inf)
    return Interval(
        jnp.where(excludes_zero, jnp.minimum(first, second), -infinity),
        jnp.where(excludes_zero, jnp.maximum(first, second), infinity),
    )


@register_pytree_node_class
class TaylorModel:
    """Dense degree-2 polynomial plus a source-independent interval remainder.

    Coefficient arrays have shapes ``value_shape``, ``value_shape + (n,)``,
    and ``value_shape + (n, n)``.  The remainder has ``value_shape`` and the
    implicit source domain is always ``[-1, 1]^n``.  Every nonconstant model
    participating in one expression must descend from the same normalized
    seed; source identity is not tracked or merged by this experimental type.
    """

    def __init__(
        self,
        constant,
        linear,
        quadratic,
        remainder: Interval,
    ):
        self.constant = jnp.asarray(constant)
        self.linear = jnp.asarray(linear)
        self.quadratic = jnp.asarray(quadratic)
        self.remainder = remainder
        if (
            self.linear.ndim != self.constant.ndim + 1
            or self.linear.shape[:-1] != self.constant.shape
        ):
            raise ValueError(
                "linear shape must be value shape + (source_size,); got "
                f"{self.linear.shape} for value shape {self.constant.shape}"
            )
        n = self.linear.shape[-1]
        if self.quadratic.shape != self.constant.shape + (n, n):
            raise ValueError(
                "quadratic shape must be value shape + (source_size, source_size); "
                f"got {self.quadratic.shape}"
            )
        if remainder.shape != self.constant.shape:
            raise ValueError("Taylor-model remainder must have the value shape.")

    def tree_flatten(self):
        return (
            (
                self.constant,
                self.linear,
                self.quadratic,
                self.remainder,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        n = self.source_size
        return TaylorModel(
            self.constant.reshape(*shape),
            self.linear.reshape(*shape, n),
            self.quadratic.reshape(*shape, n, n),
            Interval(
                self.remainder.lower.reshape(*shape),
                self.remainder.upper.reshape(*shape),
            ),
        )

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        return _value_transpose(self, axes)

    def __getitem__(self, index):
        return TaylorModel(
            self.constant[index],
            self.linear[index],
            self.quadratic[index],
            self.remainder[index],
        )

    def __add__(self, other):
        return _add(self, other)

    def __radd__(self, other):
        return _add(other, self)

    def __sub__(self, other):
        return _sub(self, other)

    def __rsub__(self, other):
        return _sub(other, self)

    def __neg__(self):
        return _neg(self)

    def __mul__(self, other):
        return _mul(self, other)

    def __rmul__(self, other):
        return _mul(other, self)

    def __truediv__(self, other):
        return _div(self, other)

    def __rtruediv__(self, other):
        return _div(other, self)


def constant_taylor_model(value, source_size) -> TaylorModel:
    """Lift a point value into a normalized source space of static size."""

    value = jnp.asarray(value)
    if isinstance(source_size, TaylorModel):
        source_size = source_size.source_size
    n = operator.index(source_size)
    if n < 0:
        raise ValueError("Taylor-model source size must be nonnegative.")
    dtype = jnp.result_type(value, float)
    value = value.astype(dtype)
    return TaylorModel(
        value,
        jnp.zeros(value.shape + (n,), dtype=dtype),
        jnp.zeros(value.shape + (n, n), dtype=dtype),
        _zero_interval(value),
    )


def interval_taylor_model(lower, upper, like: TaylorModel) -> TaylorModel:
    """Represent an interval with zero retained polynomial in ``like``'s space."""

    return _interval_only(lower, upper, like)


def append_taylor_sources(x: TaylorModel, count: int) -> TaylorModel:
    """Append normalized sources with exactly zero influence.

    The existing sources remain the leading coordinates and the appended
    coordinates are ordered last.  The represented pointwise fibers are
    unchanged for every value of the new sources in ``[-1, 1]``.
    """

    count = operator.index(count)
    if count < 0:
        raise ValueError("appended Taylor source count must be nonnegative")
    padding = ((0, 0),) * x.ndim
    return TaylorModel(
        x.constant,
        jnp.pad(x.linear, padding + ((0, count),)),
        jnp.pad(x.quadratic, padding + ((0, count), (0, count))),
        x.remainder,
    )


def lift_taylor_remainder_to_sources(x: TaylorModel) -> TaylorModel:
    """Replace a Cartesian interval remainder by independent unit sources.

    One source is appended per flattened output component in row-major order.
    For component ``i``, source ``x.source_size + i`` carries the corresponding
    remainder radius.  The returned model has zero interval remainder and
    represents exactly the complete Cartesian fiber of ``x``.
    """

    component_count = x.size
    lifted = append_taylor_sources(x, component_count)
    midpoint = 0.5 * (x.remainder.lower + x.remainder.upper)
    radius = 0.5 * (x.remainder.upper - x.remainder.lower)
    flat_linear = lifted.linear.reshape((component_count, lifted.source_size))
    component_indices = jnp.arange(component_count)
    flat_linear = flat_linear.at[
        component_indices, x.source_size + component_indices
    ].set(radius.reshape(-1))
    constant = x.constant + midpoint
    return TaylorModel(
        constant,
        flat_linear.reshape(x.shape + (lifted.source_size,)),
        lifted.quadratic,
        _zero_interval(constant),
    )


def marginalize_taylor_sources(
    x: TaylorModel, keep_source_count: int
) -> TaylorModel:
    """Eliminate trailing sources into the interval remainder.

    The leading ``keep_source_count`` sources remain in the retained
    polynomial.  Every linear or quadratic monomial involving an eliminated
    source is bounded termwise on the normalized box and added to the existing
    remainder.  Diagonal eliminated squares use ``eta**2 in [0, 1]``; mixed
    monomials use ``[-1, 1]``.
    """

    keep_source_count = operator.index(keep_source_count)
    if not 0 <= keep_source_count <= x.source_size:
        raise ValueError(
            "kept Taylor source count must lie in [0, source_size]; got "
            f"{keep_source_count} for {x.source_size}"
        )
    symmetric = 0.5 * (x.quadratic + jnp.swapaxes(x.quadratic, -1, -2))
    eliminated_linear_radius = jnp.sum(
        jnp.abs(x.linear[..., keep_source_count:]), axis=-1
    )
    mixed_radius = jnp.sum(
        jnp.abs(symmetric[..., :keep_source_count, keep_source_count:]),
        axis=(-2, -1),
    )
    eliminated_quadratic = symmetric[
        ..., keep_source_count:, keep_source_count:
    ]
    diagonal = 0.5 * jnp.diagonal(
        eliminated_quadratic, axis1=-2, axis2=-1
    )
    diagonal_lower = jnp.sum(jnp.minimum(diagonal, 0), axis=-1)
    diagonal_upper = jnp.sum(jnp.maximum(diagonal, 0), axis=-1)
    eliminated_count = x.source_size - keep_source_count
    upper_mask = jnp.triu(
        jnp.ones((eliminated_count, eliminated_count), dtype=bool), k=1
    )
    off_diagonal_radius = jnp.sum(
        jnp.where(upper_mask, jnp.abs(eliminated_quadratic), 0),
        axis=(-2, -1),
    )
    symmetric_radius = (
        eliminated_linear_radius + mixed_radius + off_diagonal_radius
    )
    return TaylorModel(
        x.constant,
        x.linear[..., :keep_source_count],
        x.quadratic[..., :keep_source_count, :keep_source_count],
        Interval(
            x.remainder.lower + diagonal_lower - symmetric_radius,
            x.remainder.upper + diagonal_upper + symmetric_radius,
        ),
    )


def inflate_taylor_remainder(
    x: TaylorModel, relative=0.0, absolute=1e-14
) -> TaylorModel:
    """Inflate only the interval remainder while preserving its midpoint."""

    midpoint = 0.5 * (x.remainder.lower + x.remainder.upper)
    radius = 0.5 * (x.remainder.upper - x.remainder.lower)
    new_radius = (1.0 + jnp.asarray(relative)) * radius + jnp.asarray(absolute)
    return TaylorModel(
        x.constant,
        x.linear,
        x.quadratic,
        Interval(midpoint - new_radius, midpoint + new_radius),
    )


def select_taylor_model(mask, on_true: TaylorModel, on_false: TaylorModel):
    """Select packed Taylor components with a point-valued JAX mask."""

    return _where_model(mask, on_true, on_false)


def taylor_hull(first: TaylorModel, second: TaylorModel) -> TaylorModel:
    """Return a pointwise-sound tube hull over common normalized sources.

    The first retained polynomial is the common polynomial.  The second
    polynomial's difference from it is ranged termwise and combined with the
    second remainder.  Taking the interval union with the first remainder
    contains both input fibers at every shared source without a sampled or
    width-based polynomial choice.
    """

    _assert_compatible(first, second)
    shape = jnp.broadcast_shapes(first.shape, second.shape)
    first = _broadcast_model(first, shape)
    second = _broadcast_model(second, shape)
    difference = TaylorModel(
        second.constant - first.constant,
        second.linear - first.linear,
        second.quadratic - first.quadratic,
        _zero_interval(first.constant),
    )
    difference_range = polynomial_range(difference)
    return TaylorModel(
        first.constant,
        first.linear,
        first.quadratic,
        Interval(
            jnp.minimum(
                first.remainder.lower,
                difference_range.lower + second.remainder.lower,
            ),
            jnp.maximum(
                first.remainder.upper,
                difference_range.upper + second.remainder.upper,
            ),
        ),
    )


def taylor_endpoint_models(x: TaylorModel) -> tuple[TaylorModel, TaylorModel]:
    """Return exact lower and upper endpoint polynomials of a Taylor tube."""

    zero = _zero_interval(x.constant)
    return (
        TaylorModel(
            x.constant + x.remainder.lower,
            x.linear,
            x.quadratic,
            zero,
        ),
        TaylorModel(
            x.constant + x.remainder.upper,
            x.linear,
            x.quadratic,
            zero,
        ),
    )


def taylor_inclusion_margins(
    outer: TaylorModel, inner: TaylorModel
) -> tuple[jax.Array, jax.Array]:
    """Bound pointwise lower and upper margins for ``inner in outer``.

    The returned arrays bound the infima of ``inner_lower - outer_lower`` and
    ``outer_upper - inner_upper`` over the common normalized source box.
    Positive values in every component certify strict pointwise containment.
    """

    _assert_compatible(outer, inner)
    shape = jnp.broadcast_shapes(outer.shape, inner.shape)
    outer = _broadcast_model(outer, shape)
    inner = _broadcast_model(inner, shape)
    polynomial_difference = TaylorModel(
        inner.constant - outer.constant,
        inner.linear - outer.linear,
        inner.quadratic - outer.quadratic,
        _zero_interval(outer.constant),
    )
    lower_margin = (
        polynomial_range(polynomial_difference).lower
        + inner.remainder.lower
        - outer.remainder.lower
    )
    upper_margin = (
        polynomial_range(-polynomial_difference).lower
        + outer.remainder.upper
        - inner.remainder.upper
    )
    return lower_margin, upper_margin


def normalized_taylor_seed(lower, upper) -> TaylorModel:
    """Normalize one physical box; poison numerically invalid endpoints."""

    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    if lower.shape != upper.shape:
        raise ValueError("Seed endpoints must have equal shapes.")
    if lower.ndim != 1 or lower.size == 0:
        raise ValueError(
            "The experimental seed requires one nonempty vector input."
        )
    numerically_valid = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper))
    numerically_valid &= jnp.all(lower <= upper)
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    n = center.size
    dtype = jnp.result_type(center, float)
    poison = jnp.asarray(jnp.nan, dtype=dtype)
    center = jnp.where(numerically_valid, center, poison)
    linear = jnp.where(
        numerically_valid,
        jnp.diag(radius.astype(dtype)),
        poison,
    )
    return TaylorModel(
        center.astype(dtype),
        linear,
        jnp.zeros((n, n, n), dtype=dtype),
        _zero_interval(center.astype(dtype)),
    )


def _assert_compatible(x: TaylorModel, y: TaylorModel) -> None:
    if x.source_size != y.source_size:
        raise ValueError("Taylor models have different source dimensions.")


def _template(*values) -> TaylorModel:
    for value in values:
        if isinstance(value, TaylorModel):
            return value
    raise TypeError("A Taylor-model operand is required.")


def _promote(value, like: TaylorModel) -> TaylorModel:
    if isinstance(value, TaylorModel):
        _assert_compatible(value, like)
        return value
    if isinstance(value, Interval):
        center = (value.lower + value.upper) / 2
        promoted = constant_taylor_model(center, like)
        promoted.remainder = Interval(value.lower - center, value.upper - center)
        return promoted
    return constant_taylor_model(value, like)


def _new(c, a, q, r) -> TaylorModel:
    return TaylorModel(c, a, q, r)


def _broadcast_model(x: TaylorModel, shape) -> TaylorModel:
    n = x.source_size
    return _new(
        jnp.broadcast_to(x.constant, shape),
        jnp.broadcast_to(x.linear, shape + (n,)),
        jnp.broadcast_to(x.quadratic, shape + (n, n)),
        Interval(
            jnp.broadcast_to(x.remainder.lower, shape),
            jnp.broadcast_to(x.remainder.upper, shape),
        ),
    )


def _interval_only(lower, upper, like: TaylorModel) -> TaylorModel:
    """Represent a value only by a source-independent interval remainder."""

    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    shape = jnp.broadcast_shapes(like.shape, lower.shape, upper.shape)
    like = _broadcast_model(like, shape)
    return _new(
        jnp.zeros_like(like.constant),
        jnp.zeros_like(like.linear),
        jnp.zeros_like(like.quadratic),
        Interval(jnp.broadcast_to(lower, shape), jnp.broadcast_to(upper, shape)),
    )


def _top_like(like: TaylorModel) -> TaylorModel:
    infinity = jnp.full_like(like.constant, jnp.inf)
    return _interval_only(-infinity, infinity, like)


def _component_ranges(x: TaylorModel) -> tuple[Interval, Interval]:
    """Return termwise ranges over the implicit normalized unit source box."""

    linear_radius = jnp.sum(jnp.abs(x.linear), axis=-1)
    linear = Interval(-linear_radius, linear_radius)

    diagonal = 0.5 * jnp.diagonal(x.quadratic, axis1=-2, axis2=-1)
    diagonal_lower = jnp.sum(jnp.minimum(diagonal, 0), axis=-1)
    diagonal_upper = jnp.sum(jnp.maximum(diagonal, 0), axis=-1)

    symmetric = 0.5 * (x.quadratic + jnp.swapaxes(x.quadratic, -1, -2))
    upper_mask = jnp.triu(
        jnp.ones((x.source_size, x.source_size), dtype=bool), k=1
    )
    off_diagonal_radius = jnp.sum(
        jnp.where(upper_mask, jnp.abs(symmetric), 0), axis=(-2, -1)
    )
    quadratic = Interval(
        diagonal_lower - off_diagonal_radius,
        diagonal_upper + off_diagonal_radius,
    )
    return linear, quadratic


def polynomial_range(x: TaylorModel) -> Interval:
    linear, quadratic = _component_ranges(x)
    return _iadd(Interval(x.constant, x.constant), _iadd(linear, quadratic))


def taylor_range(x: TaylorModel) -> Interval:
    """Termwise interval range of the represented Taylor model."""

    return _iadd(polynomial_range(x), x.remainder)


def taylor_to_affine(x: TaylorModel) -> AffineBound:
    """Keep the linear polynomial and absorb quadratic/remainder into biases.

    This conversion preserves the termwise global interval produced by
    :func:`taylor_range`, but it discards the pointwise shape of the quadratic
    polynomial.  In particular, a model with a zero linear coefficient becomes
    a pair of constant affine planes even when its quadratic term varies
    strongly over the source box.
    """

    _, quadratic = _component_ranges(x)
    discarded = _iadd(quadratic, x.remainder)
    return AffineBound(
        x.linear,
        x.constant + discarded.lower,
        x.linear,
        x.constant + discarded.upper,
        -jnp.ones((x.source_size,), dtype=x.dtype),
        jnp.ones((x.source_size,), dtype=x.dtype),
    )


def evaluate_polynomial(x: TaylorModel, sources) -> jax.Array:
    """Evaluate the retained polynomial (without the interval remainder)."""

    sources = jnp.asarray(sources)
    return (
        x.constant
        + jnp.einsum("...i,i->...", x.linear, sources)
        + 0.5 * jnp.einsum("...ij,i,j->...", x.quadratic, sources, sources)
    )


def _add(x, y):
    if not isinstance(x, TaylorModel) and not isinstance(y, TaylorModel):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return _iadd(_as_interval(x), _as_interval(y))
        return lax.add_p.bind(x, y)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    x, y = _broadcast_model(x, shape), _broadcast_model(y, shape)
    return _new(
        x.constant + y.constant,
        x.linear + y.linear,
        x.quadratic + y.quadratic,
        _iadd(x.remainder, y.remainder),
    )


def _neg(x):
    if isinstance(x, Interval):
        return _ineg(x)
    if not isinstance(x, TaylorModel):
        # JAX 0.8 Jaxpr literals can be ``TypedNdArray`` instances, which are
        # accepted by primitives but do not implement Python's unary ``-``.
        return lax.neg_p.bind(x)
    return _new(-x.constant, -x.linear, -x.quadratic, _ineg(x.remainder))


def _sub(x, y):
    return _add(x, _neg(y))


def _scale_model(x: TaylorModel, scale) -> TaylorModel:
    """Scale one model exactly by a raw point scalar or array."""

    scale = jnp.asarray(scale)
    shape = jnp.broadcast_shapes(x.shape, scale.shape)
    x = _broadcast_model(x, shape)
    scale = jnp.broadcast_to(scale, shape)
    return _new(
        x.constant * scale,
        x.linear * scale[..., None],
        x.quadratic * scale[..., None, None],
        _iscale(x.remainder, scale),
    )


def _mul(x, y):
    if not isinstance(x, TaylorModel) and not isinstance(y, TaylorModel):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return _imul(_as_interval(x), _as_interval(y))
        return lax.mul_p.bind(x, y)
    if isinstance(x, TaylorModel) and not isinstance(y, (TaylorModel, Interval)):
        return _scale_model(x, y)
    if isinstance(y, TaylorModel) and not isinstance(x, (TaylorModel, Interval)):
        return _scale_model(y, x)
    like = _template(x, y)
    x, y = _promote(x, like), _promote(y, like)
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    x, y = _broadcast_model(x, shape), _broadcast_model(y, shape)

    # Write each retained polynomial as P = c + l + q, where q denotes the
    # half-weighted quadratic form stored by ``quadratic``.  The product terms
    # kept through degree two are c1*c2, c1*l2+c2*l1, and
    # c1*q2+c2*q1+l1*l2.  Symmetrizing the two outer products below gives the
    # stored Q for l1*l2.
    c = x.constant * y.constant
    a = x.linear * y.constant[..., None] + y.linear * x.constant[..., None]
    q = (
        x.quadratic * y.constant[..., None, None]
        + y.quadratic * x.constant[..., None, None]
        + x.linear[..., :, None] * y.linear[..., None, :]
        + y.linear[..., :, None] * x.linear[..., None, :]
    )

    # The omitted polynomial is l1*q2 + q1*l2 + q1*q2.  Remainder interactions
    # follow directly from (P1 + R1)(P2 + R2).  Each term is intervalized
    # independently, which is deliberately simple and may be loose.
    lx, qx = _component_ranges(x)
    ly, qy = _component_ranges(y)
    overflow = _iadd(_imul(lx, qy), _iadd(_imul(qx, ly), _imul(qx, qy)))
    px, py = polynomial_range(x), polynomial_range(y)
    remainder = _iadd(
        overflow,
        _iadd(
            _imul(px, y.remainder),
            _iadd(_imul(py, x.remainder), _imul(x.remainder, y.remainder)),
        ),
    )
    return _new(c, a, q, remainder)


def _square(x: TaylorModel) -> TaylorModel:
    """Square one model with dependency-aware repeated-interval terms."""

    c = x.constant**2
    a = 2 * x.linear * x.constant[..., None]
    q = (
        2 * x.quadratic * x.constant[..., None, None]
        + 2 * x.linear[..., :, None] * x.linear[..., None, :]
    )

    # For P = c + l + q, the discarded polynomial is 2*l*q + q**2.
    # The complete repeated-operand remainder is 2*P*R + R**2.  Squares use
    # their dependent interval range rather than an independent rectangle
    # product of an interval with itself.
    linear, quadratic = _component_ranges(x)
    overflow = _iadd(
        _iscale(_imul(linear, quadratic), 2),
        _isquare(quadratic),
    )
    polynomial = polynomial_range(x)
    remainder = _iadd(
        overflow,
        _iadd(
            _iscale(_imul(polynomial, x.remainder), 2),
            _isquare(x.remainder),
        ),
    )
    return _new(c, a, q, remainder)


def _div(x, y):
    if isinstance(y, TaylorModel):
        return _mul(x, _unary_second_order(y, "reciprocal"))
    if isinstance(x, TaylorModel):
        return _scale_model(x, jnp.reciprocal(jnp.asarray(y)))
    if isinstance(x, Interval) or isinstance(y, Interval):
        return _imul(_as_interval(x), _ireciprocal(_as_interval(y)))
    return lax.div_p.bind(x, y)


def _second_order_candidate(
    x: TaylorModel, f0, f1, f2, third: Interval
) -> TaylorModel:
    """Apply the centered degree-2 formula for precomputed safe derivatives."""

    a = f1[..., None] * x.linear
    q = (
        f1[..., None, None] * x.quadratic
        + f2[..., None, None] * x.linear[..., :, None] * x.linear[..., None, :]
    )

    linear, quadratic = _component_ranges(x)
    displacement_tail = _iadd(quadratic, x.remainder)
    second_order_tail = _iadd(
        _iscale(_imul(linear, displacement_tail), 2),
        _isquare(displacement_tail),
    )
    h = _iadd(linear, displacement_tail)
    lagrange = _iscale(_imul(_ipow3(h), third), 1 / 6)
    remainder = _iadd(
        _iscale(x.remainder, f1),
        _iadd(_iscale(second_order_tail, 0.5 * f2), lagrange),
    )
    return _new(f0, a, q, remainder)


def _unary_second_order(x: TaylorModel, function: str, exponent=None) -> TaylorModel:
    """Centered order-2 unary rule with masked finite fallbacks or top."""

    c = x.constant
    input_bounds = taylor_range(x)
    # Lagrange points lie between the expansion point and represented values.
    derivative_lower = jnp.minimum(input_bounds.lower, c)
    derivative_upper = jnp.maximum(input_bounds.upper, c)

    if function == "sin":
        return _second_order_candidate(
            x,
            jnp.sin(c),
            jnp.cos(c),
            -jnp.sin(c),
            Interval(-jnp.ones_like(c), jnp.ones_like(c)),
        )
    if function == "cos":
        return _second_order_candidate(
            x,
            jnp.cos(c),
            -jnp.sin(c),
            -jnp.cos(c),
            Interval(-jnp.ones_like(c), jnp.ones_like(c)),
        )

    top = _top_like(x)
    if function == "sqrt":
        defined = input_bounds.lower >= 0
        regular_precondition = defined & (derivative_lower > 0) & jnp.isfinite(c)
        safe_c = jnp.where(regular_precondition, c, 1)
        safe_lower = jnp.where(regular_precondition, derivative_lower, 1)
        safe_upper = jnp.where(regular_precondition, derivative_upper, 1)
        root = jnp.sqrt(safe_c)
        f0, f1, f2 = root, 0.5 / root, -0.25 / (safe_c * root)
        third = Interval(
            3 / (8 * safe_upper**2.5),
            3 / (8 * safe_lower**2.5),
        )
        regular = regular_precondition & jnp.isfinite(f0)
        regular &= jnp.isfinite(f1) & jnp.isfinite(f2)
        candidate = _second_order_candidate(x, f0, f1, f2, third)

        safe_bound_lower = jnp.where(defined, input_bounds.lower, 0)
        safe_bound_upper = jnp.where(defined, input_bounds.upper, 0)
        fallback = _interval_only(
            jnp.sqrt(safe_bound_lower), jnp.sqrt(safe_bound_upper), x
        )
        return _where_model(regular, candidate, _where_model(defined, fallback, top))

    if function == "reciprocal":
        represented_excludes_zero = (input_bounds.lower > 0) | (
            input_bounds.upper < 0
        )
        derivative_excludes_zero = (derivative_lower > 0) | (
            derivative_upper < 0
        )
        regular_precondition = (
            represented_excludes_zero
            & derivative_excludes_zero
            & jnp.isfinite(c)
        )
        safe_c = jnp.where(regular_precondition, c, 1)
        safe_lower = jnp.where(regular_precondition, derivative_lower, 1)
        safe_upper = jnp.where(regular_precondition, derivative_upper, 1)
        f0, f1, f2 = 1 / safe_c, -1 / safe_c**2, 2 / safe_c**3
        third_at_lower = -6 / safe_lower**4
        third_at_upper = -6 / safe_upper**4
        third = Interval(
            jnp.minimum(third_at_lower, third_at_upper),
            jnp.maximum(third_at_lower, third_at_upper),
        )
        regular = regular_precondition & jnp.isfinite(f0)
        regular &= jnp.isfinite(f1) & jnp.isfinite(f2)
        candidate = _second_order_candidate(x, f0, f1, f2, third)
        fallback_bounds = _ireciprocal(input_bounds)
        fallback = _interval_only(
            fallback_bounds.lower, fallback_bounds.upper, x
        )
        return _where_model(
            regular,
            candidate,
            _where_model(represented_excludes_zero, fallback, top),
        )

    if function == "power":
        p = jnp.asarray(exponent)
        if p.ndim != 0:
            raise ValueError("experimental Taylor power exponent must be scalar")
        if not (
            jnp.issubdtype(p.dtype, jnp.integer)
            or jnp.issubdtype(p.dtype, jnp.floating)
        ):
            return top

        exponent_finite = jnp.isfinite(p)
        positive_segment = derivative_lower > 0
        regular_precondition = positive_segment & exponent_finite & jnp.isfinite(c)
        safe_p = jnp.where(exponent_finite, p, 1)
        safe_c = jnp.where(regular_precondition, c, 1)
        safe_lower = jnp.where(regular_precondition, derivative_lower, 1)
        safe_upper = jnp.where(regular_precondition, derivative_upper, 1)
        f0 = safe_c**safe_p
        f1 = safe_p * safe_c ** (safe_p - 1)
        f2 = safe_p * (safe_p - 1) * safe_c ** (safe_p - 2)
        third_factor = safe_p * (safe_p - 1) * (safe_p - 2)
        third_at_lower = third_factor * safe_lower ** (safe_p - 3)
        third_at_upper = third_factor * safe_upper ** (safe_p - 3)
        third = Interval(
            jnp.minimum(third_at_lower, third_at_upper),
            jnp.maximum(third_at_lower, third_at_upper),
        )
        regular = regular_precondition & jnp.isfinite(f0)
        regular &= jnp.isfinite(f1) & jnp.isfinite(f2)
        regular &= jnp.isfinite(third.lower) & jnp.isfinite(third.upper)
        candidate = _second_order_candidate(x, f0, f1, f2, third)

        nonnegative = input_bounds.lower >= 0
        safe_bound_lower = jnp.where(nonnegative, input_bounds.lower, 1)
        safe_bound_upper = jnp.where(nonnegative, input_bounds.upper, 1)
        first = safe_bound_lower**safe_p
        second = safe_bound_upper**safe_p
        fallback_lower = jnp.minimum(first, second)
        fallback_upper = jnp.maximum(first, second)
        finite_fallback = nonnegative & exponent_finite
        finite_fallback &= jnp.isfinite(fallback_lower) & jnp.isfinite(fallback_upper)
        fallback = _interval_only(fallback_lower, fallback_upper, x)
        return _where_model(
            regular, candidate, _where_model(finite_fallback, fallback, top)
        )

    raise ValueError(function)  # pragma: no cover - private static dispatch


def _sin(x, **_):
    return _unary_second_order(x, "sin") if isinstance(x, TaylorModel) else jnp.sin(x)


def _cos(x, **_):
    return _unary_second_order(x, "cos") if isinstance(x, TaylorModel) else jnp.cos(x)


def _sqrt(x, **params):
    if isinstance(x, TaylorModel):
        return _unary_second_order(x, "sqrt")
    return lax.sqrt_p.bind(x, **params)


def _pow(x, y):
    if not isinstance(x, TaylorModel) and not isinstance(y, TaylorModel):
        return lax.pow_p.bind(x, y)
    if isinstance(y, TaylorModel):
        raise NotImplementedError(
            "experimental Taylor pow requires a raw point exponent, not a "
            "Taylor-model exponent"
        )
    if not isinstance(x, TaylorModel):
        return x**y
    return _unary_second_order(x, "power", exponent=y)


def _integer_pow(x, *, y):
    if not isinstance(x, TaylorModel):
        return lax.integer_pow_p.bind(x, y=y)
    if y == 0:
        return constant_taylor_model(jnp.ones_like(x.constant), x)
    if y == 1:
        return x
    if y == 2:
        return _square(x)
    raise NotImplementedError(
        f"experimental Taylor integer_pow only supports 0, 1, 2; got {y}"
    )


def _difference_range(x, y) -> Interval:
    if isinstance(x, TaylorModel) or isinstance(y, TaylorModel):
        like = _template(x, y)
        return taylor_range(_sub(_promote(x, like), _promote(y, like)))
    if isinstance(x, Interval) or isinstance(y, Interval):
        x = x if isinstance(x, Interval) else Interval(x, x)
        y = y if isinstance(y, Interval) else Interval(y, y)
        return _isub(x, y)
    difference = x - y
    return Interval(difference, difference)


def _comparison(kind, x, y):
    bounds = _difference_range(x, y)
    if kind == "lt":
        return Interval(bounds.upper < 0, bounds.lower < 0)
    if kind == "le":
        return Interval(bounds.upper <= 0, bounds.lower <= 0)
    if kind == "gt":
        return _comparison("lt", y, x)
    if kind == "ge":
        return _comparison("le", y, x)
    if kind == "eq":
        return Interval(
            (bounds.lower == 0) & (bounds.upper == 0),
            (bounds.lower <= 0) & (bounds.upper >= 0),
        )
    if kind == "ne":
        equal = _comparison("eq", x, y)
        return Interval(~equal.upper, ~equal.lower)
    raise ValueError(kind)


def _boolean_binary(x, y, *, is_and):
    x = x if isinstance(x, Interval) else Interval(x, x)
    y = y if isinstance(y, Interval) else Interval(y, y)
    operation = jnp.logical_and if is_and else jnp.logical_or
    return Interval(operation(x.lower, y.lower), operation(x.upper, y.upper))


def _boolean_not(x):
    x = x if isinstance(x, Interval) else Interval(x, x)
    return Interval(jnp.logical_not(x.upper), jnp.logical_not(x.lower))


def _reduce_boolean(x, *, axes, is_and):
    primitive = lax.reduce_and_p if is_and else lax.reduce_or_p
    if not isinstance(x, Interval):
        return primitive.bind(x, axes=axes)
    return Interval(
        primitive.bind(x.lower, axes=axes),
        primitive.bind(x.upper, axes=axes),
    )


def _where_model(mask, yes: TaylorModel, no: TaylorModel) -> TaylorModel:
    like = _template(yes, no)
    yes, no = _promote(yes, like), _promote(no, like)
    shape = jnp.broadcast_shapes(jnp.shape(mask), yes.shape, no.shape)
    yes, no = _broadcast_model(yes, shape), _broadcast_model(no, shape)
    mask = jnp.broadcast_to(mask, shape)
    return _new(
        jnp.where(mask, yes.constant, no.constant),
        jnp.where(mask[..., None], yes.linear, no.linear),
        jnp.where(mask[..., None, None], yes.quadratic, no.quadratic),
        Interval(
            jnp.where(mask, yes.remainder.lower, no.remainder.lower),
            jnp.where(mask, yes.remainder.upper, no.remainder.upper),
        ),
    )


def _select_models(which, cases) -> TaylorModel:
    like = _template(*cases)
    promoted = [_promote(case, like) for case in cases]
    result = promoted[0]
    for index, case in enumerate(promoted[1:], start=1):
        result = _where_model(which == index, case, result)
    return result


def _hull_models(cases, selection_shape=()) -> TaylorModel:
    like = _template(*cases)
    promoted = [_promote(case, like) for case in cases]
    shape = jnp.broadcast_shapes(selection_shape, *(case.shape for case in promoted))
    promoted = [_broadcast_model(case, shape) for case in promoted]
    ranges = [taylor_range(case) for case in promoted]
    lower = jnp.min(jnp.stack([bounds.lower for bounds in ranges]), axis=0)
    upper = jnp.max(jnp.stack([bounds.upper for bounds in ranges]), axis=0)
    return _interval_only(lower, upper, promoted[0])


def _select_intervals(which, cases) -> Interval:
    shape = jnp.broadcast_shapes(
        jnp.shape(which),
        *(case.shape for case in cases),
    )
    which = jnp.broadcast_to(which, shape)
    lowers = [jnp.broadcast_to(case.lower, shape) for case in cases]
    uppers = [jnp.broadcast_to(case.upper, shape) for case in cases]
    lower, upper = lowers[0], uppers[0]
    for index, (case_lower, case_upper) in enumerate(
        zip(lowers[1:], uppers[1:]), start=1
    ):
        selected = which == index
        lower = jnp.where(selected, case_lower, lower)
        upper = jnp.where(selected, case_upper, upper)
    return Interval(lower, upper)


def _select_n(which, *cases):
    if isinstance(which, Interval):
        fixed = which.lower == which.upper
        if any(isinstance(case, TaylorModel) for case in cases):
            selected = _select_models(which.lower, cases)
            hull = _hull_models(cases, jnp.shape(fixed))
            return _where_model(fixed, selected, hull)
        intervals = [
            case if isinstance(case, Interval) else Interval(case, case)
            for case in cases
        ]
        selected = _select_intervals(which.lower, intervals)
        shape = selected.shape
        hull_lower = jnp.min(
            jnp.stack([jnp.broadcast_to(case.lower, shape) for case in intervals]),
            axis=0,
        )
        hull_upper = jnp.max(
            jnp.stack([jnp.broadcast_to(case.upper, shape) for case in intervals]),
            axis=0,
        )
        fixed = jnp.broadcast_to(fixed, shape)
        return Interval(
            jnp.where(fixed, selected.lower, hull_lower),
            jnp.where(fixed, selected.upper, hull_upper),
        )
    if not any(isinstance(case, TaylorModel) for case in cases):
        if any(isinstance(case, Interval) for case in cases):
            intervals = [
                case if isinstance(case, Interval) else Interval(case, case)
                for case in cases
            ]
            return _select_intervals(which, intervals)
        return lax.select_n(which, *cases)
    return _select_models(which, cases)


def taylor_relu(x):
    """Apply exact fixed ReLU branches and a source-preserving crossing rule."""

    if isinstance(x, Interval):
        return Interval(jnp.maximum(x.lower, 0), jnp.maximum(x.upper, 0))
    if not isinstance(x, TaylorModel):
        return jnp.maximum(x, 0)

    bounds = taylor_range(x)
    active = bounds.lower >= 0
    inactive = bounds.upper <= 0
    crossing = ~(active | inactive)
    finite_crossing = crossing & jnp.isfinite(bounds.lower)
    finite_crossing &= jnp.isfinite(bounds.upper)
    safe_lower = jnp.where(finite_crossing, bounds.lower, -1)
    safe_upper = jnp.where(finite_crossing, bounds.upper, 1)
    denominator = safe_upper - safe_lower
    slope = safe_upper / denominator
    error_upper = -safe_lower * safe_upper / denominator
    scaled = _scale_model(x, slope)
    crossing_model = _new(
        scaled.constant,
        scaled.linear,
        scaled.quadratic,
        _iadd(
            scaled.remainder,
            Interval(jnp.zeros_like(error_upper), error_upper),
        ),
    )
    fallback = _interval_only(
        jnp.maximum(bounds.lower, 0), jnp.maximum(bounds.upper, 0), x
    )
    crossing_model = _where_model(finite_crossing, crossing_model, fallback)
    zero = constant_taylor_model(jnp.zeros_like(x.constant), x)
    return _where_model(active, x, _where_model(inactive, zero, crossing_model))


def _max_or_min(x, y, *, is_max):
    # Promote an Interval to a Taylor remainder when the other operand is a
    # Taylor model.  Handling Interval first would attempt to store a
    # TaylorModel object as an Interval endpoint.
    if isinstance(x, TaylorModel) or isinstance(y, TaylorModel):
        like = _template(x, y)
        left, right = _promote(x, like), _promote(y, like)
        difference = _sub(left, right)
        bounds = taylor_range(difference)
        left_above = bounds.lower >= 0
        left_below = bounds.upper <= 0
        relu_difference = taylor_relu(difference)
        crossing = (
            _add(right, relu_difference)
            if is_max
            else _sub(left, relu_difference)
        )
        if is_max:
            return _where_model(
                left_above,
                left,
                _where_model(left_below, right, crossing),
            )
        return _where_model(
            left_above,
            right,
            _where_model(left_below, left, crossing),
        )
    if isinstance(x, Interval) or isinstance(y, Interval):
        x = x if isinstance(x, Interval) else Interval(x, x)
        y = y if isinstance(y, Interval) else Interval(y, y)
        operation = jnp.maximum if is_max else jnp.minimum
        return Interval(operation(x.lower, y.lower), operation(x.upper, y.upper))
    return jnp.maximum(x, y) if is_max else jnp.minimum(x, y)


def _abs(x):
    if not isinstance(x, TaylorModel):
        return jnp.abs(x)
    return _max_or_min(x, -x, is_max=True)


def _apply_unary_value_linear(primitive, x: TaylorModel, **params) -> TaylorModel:
    if isinstance(x, Interval):
        return Interval(
            primitive.bind(x.lower, **params),
            primitive.bind(x.upper, **params),
        )
    if not isinstance(x, TaylorModel):
        return primitive.bind(x, **params)
    apply_a = lambda value: jax.vmap(
        lambda part: primitive.bind(part, **params), in_axes=-1, out_axes=-1
    )(value)
    apply_q = lambda value: jax.vmap(
        lambda row: jax.vmap(
            lambda part: primitive.bind(part, **params), in_axes=-1, out_axes=-1
        )(row),
        in_axes=-1,
        out_axes=-1,
    )(value)
    return _new(
        primitive.bind(x.constant, **params),
        apply_a(x.linear),
        apply_q(x.quadratic),
        Interval(
            primitive.bind(x.remainder.lower, **params),
            primitive.bind(x.remainder.upper, **params),
        ),
    )


def _value_transpose(x: TaylorModel, axes) -> TaylorModel:
    n = x.ndim
    axes = tuple(axes)
    return _new(
        jnp.transpose(x.constant, axes),
        jnp.transpose(x.linear, axes + (n,)),
        jnp.transpose(x.quadratic, axes + (n, n + 1)),
        Interval(
            jnp.transpose(x.remainder.lower, axes),
            jnp.transpose(x.remainder.upper, axes),
        ),
    )


def _concatenate(*values, **params):
    if not any(isinstance(value, TaylorModel) for value in values):
        if any(isinstance(value, Interval) for value in values):
            intervals = [
                value if isinstance(value, Interval) else Interval(value, value)
                for value in values
            ]
            dimension = params["dimension"]
            return Interval(
                jnp.concatenate([value.lower for value in intervals], axis=dimension),
                jnp.concatenate([value.upper for value in intervals], axis=dimension),
            )
        return lax.concatenate_p.bind(*values, **params)
    like = _template(*values)
    values = [_promote(value, like) for value in values]
    dimension = params["dimension"]
    return _new(
        jnp.concatenate([value.constant for value in values], axis=dimension),
        jnp.concatenate([value.linear for value in values], axis=dimension),
        jnp.concatenate([value.quadratic for value in values], axis=dimension),
        Interval(
            jnp.concatenate(
                [value.remainder.lower for value in values], axis=dimension
            ),
            jnp.concatenate(
                [value.remainder.upper for value in values], axis=dimension
            ),
        ),
    )


def _gather(operand, indices, **params):
    if isinstance(indices, (TaylorModel, Interval)):
        raise NotImplementedError("experimental Taylor gather requires point indices")
    if not isinstance(operand, TaylorModel):
        if isinstance(operand, Interval):
            return Interval(
                lax.gather_p.bind(operand.lower, indices, **params),
                lax.gather_p.bind(operand.upper, indices, **params),
            )
        return lax.gather_p.bind(operand, indices, **params)
    return _apply_unary_value_linear(
        _BoundPrimitive(lambda value, **_: lax.gather_p.bind(value, indices, **params)),
        operand,
    )


class _BoundPrimitive:
    """Adapter exposing a small callable through the primitive ``bind`` API."""

    def __init__(self, function):
        self.function = function

    def bind(self, value, **params):
        return self.function(value, **params)


def _scatter_linear(primitive, operand, indices, updates, **params):
    if isinstance(indices, (TaylorModel, Interval)):
        raise NotImplementedError("experimental Taylor scatter requires point indices")
    if not isinstance(operand, TaylorModel) and not isinstance(updates, TaylorModel):
        return primitive.bind(operand, indices, updates, **params)
    like = _template(operand, updates)
    operand, updates = _promote(operand, like), _promote(updates, like)

    def apply(left, right):
        return primitive.bind(left, indices, right, **params)

    apply_a = lambda left, right: jax.vmap(apply, in_axes=(-1, -1), out_axes=-1)(
        left, right
    )
    apply_q = lambda left, right: jax.vmap(
        lambda lrow, rrow: jax.vmap(apply, in_axes=(-1, -1), out_axes=-1)(lrow, rrow),
        in_axes=(-1, -1),
        out_axes=-1,
    )(left, right)
    return _new(
        apply(operand.constant, updates.constant),
        apply_a(operand.linear, updates.linear),
        apply_q(operand.quadratic, updates.quadratic),
        Interval(
            apply(operand.remainder.lower, updates.remainder.lower),
            apply(operand.remainder.upper, updates.remainder.upper),
        ),
    )


def _reduce_sum(x, *, axes, **params):
    if isinstance(x, Interval):
        return Interval(
            lax.reduce_sum_p.bind(x.lower, axes=axes, **params),
            lax.reduce_sum_p.bind(x.upper, axes=axes, **params),
        )
    if not isinstance(x, TaylorModel):
        return lax.reduce_sum_p.bind(x, axes=axes, **params)
    return _apply_unary_value_linear(lax.reduce_sum_p, x, axes=axes, **params)


def _moveaxis(x: TaylorModel, source, destination) -> TaylorModel:
    source, destination = tuple(source), tuple(destination)
    return _new(
        jnp.moveaxis(x.constant, source, destination),
        jnp.moveaxis(x.linear, source, destination),
        jnp.moveaxis(x.quadratic, source, destination),
        Interval(
            jnp.moveaxis(x.remainder.lower, source, destination),
            jnp.moveaxis(x.remainder.upper, source, destination),
        ),
    )


def _split(x, *, sizes, axis):
    if not isinstance(x, TaylorModel):
        return lax.split_p.bind(x, sizes=sizes, axis=axis)
    outputs = []
    start = 0
    for size in sizes:
        stop = start + operator.index(size)

        def value_slice(value):
            return lax.slice_in_dim(value, start, stop, axis=axis)

        outputs.append(
            TaylorModel(
                value_slice(x.constant),
                value_slice(x.linear),
                value_slice(x.quadratic),
                Interval(
                    value_slice(x.remainder.lower),
                    value_slice(x.remainder.upper),
                ),
            )
        )
        start = stop
    return outputs


def _dot_general(a, b, **params):
    if not isinstance(a, TaylorModel) and not isinstance(b, TaylorModel):
        return lax.dot_general_p.bind(a, b, **params)
    like = _template(a, b)
    a, b = _promote(a, like), _promote(b, like)
    (ac, bc), (ab, bb) = params["dimension_numbers"]
    a = _moveaxis(a, ab + ac, tuple(range(len(ab) + len(ac))))
    b = _moveaxis(b, bb + bc, tuple(range(len(bb) + len(bc))))
    batch_n, contract_n = len(ab), len(ac)
    a_free = a.shape[batch_n + contract_n :]
    b_free = b.shape[batch_n + contract_n :]
    a = a.reshape(a.shape + (1,) * len(b_free))
    b = b.reshape(b.shape[: batch_n + contract_n] + (1,) * len(a_free) + b_free)
    products = _mul(a, b)
    axes = tuple(range(batch_n, batch_n + contract_n))
    return _reduce_sum(products, axes=axes) if axes else products


def _convert_element_type(x, **params):
    new_dtype = params["new_dtype"]
    if new_dtype == jnp.bool_:
        if isinstance(x, TaylorModel):
            bounds = taylor_range(x)
        elif isinstance(x, Interval):
            bounds = x
        else:
            return lax.convert_element_type_p.bind(x, **params)
        definitely_nonzero = (bounds.lower > 0) | (bounds.upper < 0)
        possibly_nonzero = ~((bounds.lower == 0) & (bounds.upper == 0))
        return Interval(definitely_nonzero, possibly_nonzero)
    if isinstance(x, Interval):
        return Interval(
            lax.convert_element_type_p.bind(x.lower, **params),
            lax.convert_element_type_p.bind(x.upper, **params),
        )
    if not isinstance(x, TaylorModel):
        return lax.convert_element_type_p.bind(x, **params)
    if not jnp.issubdtype(new_dtype, jnp.floating):
        raise NotImplementedError(
            "experimental Taylor conversion to a non-floating dtype is unsupported"
        )
    return _apply_unary_value_linear(lax.convert_element_type_p, x, **params)


def _copy_or_stop(primitive, x):
    if isinstance(x, (TaylorModel, Interval)):
        return _apply_unary_value_linear(primitive, x)
    return primitive.bind(x)


taylor_inclusion_registry: dict[Any, Callable[..., Any]] = {}


def register_taylor_rule(primitive, rule: Callable[..., Any]) -> None:
    """Register a local rule, including rules for ``custom_if`` primitives."""

    taylor_inclusion_registry[primitive] = rule


taylor_inclusion_registry.update(
    {
        lax.add_p: _add,
        ad_util.add_any_p: _add,
        lax.sub_p: _sub,
        lax.neg_p: _neg,
        lax.mul_p: _mul,
        lax.div_p: _div,
        lax.sin_p: _sin,
        lax.cos_p: _cos,
        lax.sqrt_p: _sqrt,
        lax.pow_p: _pow,
        lax.integer_pow_p: _integer_pow,
        lax.abs_p: _abs,
        lax.max_p: lambda x, y: _max_or_min(x, y, is_max=True),
        lax.min_p: lambda x, y: _max_or_min(x, y, is_max=False),
        lax.lt_p: lambda x, y: _comparison("lt", x, y),
        lax.le_p: lambda x, y: _comparison("le", x, y),
        lax.gt_p: lambda x, y: _comparison("gt", x, y),
        lax.ge_p: lambda x, y: _comparison("ge", x, y),
        lax.eq_p: lambda x, y: _comparison("eq", x, y),
        lax.ne_p: lambda x, y: _comparison("ne", x, y),
        lax.and_p: lambda x, y: _boolean_binary(x, y, is_and=True),
        lax.or_p: lambda x, y: _boolean_binary(x, y, is_and=False),
        lax.not_p: _boolean_not,
        lax.select_n_p: _select_n,
        lax.convert_element_type_p: _convert_element_type,
        lax.copy_p: lambda x: _copy_or_stop(lax.copy_p, x),
        lax.stop_gradient_p: lambda x: _copy_or_stop(lax.stop_gradient_p, x),
        lax.reshape_p: lambda x, **params: _apply_unary_value_linear(
            lax.reshape_p, x, **params
        ),
        lax.slice_p: lambda x, **params: _apply_unary_value_linear(
            lax.slice_p, x, **params
        ),
        lax.squeeze_p: lambda x, **params: _apply_unary_value_linear(
            lax.squeeze_p, x, **params
        ),
        lax.transpose_p: lambda x, **params: _apply_unary_value_linear(
            lax.transpose_p, x, **params
        ),
        lax.broadcast_in_dim_p: lambda x, **params: _apply_unary_value_linear(
            lax.broadcast_in_dim_p, x, **params
        ),
        lax.concatenate_p: _concatenate,
        lax.gather_p: _gather,
        lax.scatter_p: lambda *args, **params: _scatter_linear(
            lax.scatter_p, *args, **params
        ),
        lax.scatter_add_p: lambda *args, **params: _scatter_linear(
            lax.scatter_add_p, *args, **params
        ),
        lax.reduce_sum_p: _reduce_sum,
        lax.reduce_and_p: lambda x, *, axes: _reduce_boolean(x, axes=axes, is_and=True),
        lax.reduce_or_p: lambda x, *, axes: _reduce_boolean(x, axes=axes, is_and=False),
        lax.dot_general_p: _dot_general,
    }
)

if hasattr(lax, "split_p"):
    taylor_inclusion_registry[lax.split_p] = _split


def _jit_rule(*args, **params):
    closed = params.pop("jaxpr")
    if isinstance(closed, jax.extend.core.ClosedJaxpr):
        return taylor_jaxpr(closed.jaxpr, closed.consts, *args)
    return taylor_jaxpr(closed, [], *args)


try:
    taylor_inclusion_registry[jax._src.pjit.jit_p] = _jit_rule
except AttributeError:  # pragma: no cover - JAX version compatibility
    pass


def _is_abstract(value):
    return isinstance(value, (TaylorModel, Interval))


def taylor_jaxpr(jaxpr, consts, *args) -> list[Any]:
    """Evaluate one Jaxpr using the deliberately small Taylor registry.

    Pure equations outside the backward slice of the requested outputs are
    skipped.  Effectful equations are retained.  This matters for MuJoDiCo,
    whose context Jaxpr contains a Cholesky factor that is not used by the
    predictor residual and therefore is not part of the operation surface of
    this experiment.
    """

    env = {}

    # ``filter_make_jaxpr`` can retain equations whose values are not used by
    # the requested outputs.  In particular, MuJoDiCo's context construction
    # contains a source-dependent Cholesky factor that is irrelevant to the
    # predictor residual.  Do the same output liveness pruning used by the
    # operation audit so unsupported primitives in dead subgraphs are not
    # interpreted as requirements of the transformed function.
    live_variables = {
        variable for variable in jaxpr.outvars if not isinstance(variable, Literal)
    }
    live_equations = set()
    for index in range(len(jaxpr.eqns) - 1, -1, -1):
        equation = jaxpr.eqns[index]
        output_is_live = any(
            variable in live_variables for variable in equation.outvars
        )
        if output_is_live or equation.effects:
            live_equations.add(index)
            live_variables.update(
                variable
                for variable in equation.invars
                if not isinstance(variable, Literal)
            )

    def read(variable):
        return variable.val if isinstance(variable, Literal) else env[variable]

    def write(variable, value):
        env[variable] = value

    for variable, value in zip(jaxpr.constvars, consts):
        write(variable, value)
    for variable, value in zip(jaxpr.invars, args):
        write(variable, value)
    for index, equation in enumerate(jaxpr.eqns):
        if index not in live_equations:
            continue
        subfuns, bind_params = equation.primitive.get_bind_params(equation.params)
        inputs = [read(variable) for variable in equation.invars]
        if any(_is_abstract(value) for value in inputs):
            try:
                rule = taylor_inclusion_registry[equation.primitive]
            except KeyError as error:
                raise NotImplementedError(
                    "experimental Taylor primitive unsupported: "
                    f"{equation.primitive.name}"
                ) from error
            answer = rule(*subfuns, *inputs, **bind_params)
        else:
            answer = equation.primitive.bind(*subfuns, *inputs, **bind_params)
        if equation.primitive.multiple_results:
            for variable, value in zip(equation.outvars, answer):
                write(variable, value)
        else:
            write(equation.outvars[0], answer)
    return [read(variable) for variable in jaxpr.outvars]


def tmif(function: Callable[..., Any]) -> Callable[..., Any]:
    """Create an eager experimental Taylor-model transform for ``function``.

    The current wrapper is intended for positional Taylor-model inputs and the
    flat array output used by the hinge residual.  It is not a replacement for
    the general ``affif`` API and does not alter affine inclusion behavior.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        abstract = [
            value
            for value in jax.tree_util.tree_leaves(args, is_leaf=_is_abstract)
            if isinstance(value, TaylorModel)
        ]
        if not abstract:
            return function(*args, **kwargs)
        like = abstract[0]
        for value in abstract[1:]:
            _assert_compatible(value, like)
        trace_args = jax.tree_util.tree_map(
            lambda value: value.constant if isinstance(value, TaylorModel) else value,
            args,
            is_leaf=_is_abstract,
        )
        trace_kwargs = jax.tree_util.tree_map(
            lambda value: value.constant if isinstance(value, TaylorModel) else value,
            kwargs,
            is_leaf=_is_abstract,
        )
        traced = lambda *positional: function(*positional, **trace_kwargs)
        closed = eqx.filter_make_jaxpr(traced)(*trace_args)[0]
        flat_args = [
            value
            for value in jax.tree_util.tree_leaves(args, is_leaf=_is_abstract)
            if isinstance(value, TaylorModel) or eqx.is_array(value)
        ]
        outputs = taylor_jaxpr(closed.jaxpr, closed.literals, *flat_args)
        return outputs[0] if len(outputs) == 1 else outputs

    return wrapped
