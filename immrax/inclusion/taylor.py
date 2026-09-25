"""Experimental degree-2 Taylor models over one shared source box.

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
termwise polynomial ranges, sign-certified max/ReLU branches only, division
only when the denominator range excludes zero, eager-only branch decisions,
one flat transformed output, and an incomplete primitive surface.  Custom
primitive rules live in a mutable experimental registry and must be installed
explicitly by a caller such as the MuJoDiCo diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
    try:
        excludes_zero = bool(np.all(np.asarray((x.lower > 0) | (x.upper < 0))))
    except (TypeError, jax.errors.TracerArrayConversionError):
        excludes_zero = False
    if not excludes_zero:
        raise NotImplementedError(
            "experimental interval reciprocal denominator contains zero"
        )
    first, second = 1 / x.lower, 1 / x.upper
    return Interval(jnp.minimum(first, second), jnp.maximum(first, second))


@register_pytree_node_class
class TaylorModel:
    """Dense degree-2 polynomial plus a source-independent interval remainder.

    Coefficient arrays have shapes ``value_shape``, ``value_shape + (n,)``,
    and ``value_shape + (n, n)``.  The remainder has ``value_shape``.  Every
    model participating in one expression must use the same source domain.
    """

    def __init__(
        self,
        constant,
        linear,
        quadratic,
        remainder: Interval,
        domain_lower,
        domain_upper,
    ):
        self.constant = jnp.asarray(constant)
        self.linear = jnp.asarray(linear)
        self.quadratic = jnp.asarray(quadratic)
        self.remainder = remainder
        self.domain_lower = jnp.asarray(domain_lower).reshape(-1)
        self.domain_upper = jnp.asarray(domain_upper).reshape(-1)
        n = self.domain_lower.size
        if self.domain_upper.shape != self.domain_lower.shape:
            raise ValueError("Taylor-model source bounds must have equal shapes.")
        if self.linear.shape != self.constant.shape + (n,):
            raise ValueError(
                "linear shape must be value shape + (source_size,); got "
                f"{self.linear.shape} for {self.constant.shape} and n={n}"
            )
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
                self.domain_lower,
                self.domain_upper,
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
        return self.domain_lower.size

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
            self.domain_lower,
            self.domain_upper,
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
            self.domain_lower,
            self.domain_upper,
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


def constant_taylor_model(value, domain_lower, domain_upper) -> TaylorModel:
    """Lift a point value into an existing normalized source domain."""

    value = jnp.asarray(value)
    domain_lower = jnp.asarray(domain_lower).reshape(-1)
    domain_upper = jnp.asarray(domain_upper).reshape(-1)
    n = domain_lower.size
    dtype = jnp.result_type(value, float)
    value = value.astype(dtype)
    return TaylorModel(
        value,
        jnp.zeros(value.shape + (n,), dtype=dtype),
        jnp.zeros(value.shape + (n, n), dtype=dtype),
        _zero_interval(value),
        domain_lower,
        domain_upper,
    )


def normalized_taylor_seed(lower, upper) -> TaylorModel:
    """Represent one physical box using shared normalized sources."""

    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    if lower.shape != upper.shape:
        raise ValueError("Seed endpoints must have equal shapes.")
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    n = center.size
    if center.ndim != 1:
        raise ValueError("The experimental seed currently requires one vector input.")
    dtype = jnp.result_type(center, float)
    return TaylorModel(
        center.astype(dtype),
        jnp.diag(radius.astype(dtype)),
        jnp.zeros((n, n, n), dtype=dtype),
        _zero_interval(center.astype(dtype)),
        -jnp.ones((n,), dtype=dtype),
        jnp.ones((n,), dtype=dtype),
    )


def _assert_compatible(x: TaylorModel, y: TaylorModel) -> None:
    if x.source_size != y.source_size:
        raise ValueError("Taylor models have different source dimensions.")
    try:
        compatible = np.array_equal(
            np.asarray(x.domain_lower), np.asarray(y.domain_lower)
        )
        compatible &= np.array_equal(
            np.asarray(x.domain_upper), np.asarray(y.domain_upper)
        )
    except (TypeError, jax.errors.TracerArrayConversionError):
        compatible = True
    if not compatible:
        raise ValueError("Taylor models have incompatible source domains.")


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
        promoted = constant_taylor_model(center, like.domain_lower, like.domain_upper)
        promoted.remainder = Interval(value.lower - center, value.upper - center)
        return promoted
    return constant_taylor_model(value, like.domain_lower, like.domain_upper)


def _new(c, a, q, r, like: TaylorModel) -> TaylorModel:
    return TaylorModel(c, a, q, r, like.domain_lower, like.domain_upper)


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
        x,
    )


def _component_ranges(x: TaylorModel) -> tuple[Interval, Interval]:
    """Return termwise ranges of the linear and quadratic components."""

    dl, du = x.domain_lower, x.domain_upper
    a_pos, a_neg = jnp.maximum(x.linear, 0), jnp.minimum(x.linear, 0)
    linear = Interval(
        jnp.sum(a_pos * dl + a_neg * du, axis=-1),
        jnp.sum(a_pos * du + a_neg * dl, axis=-1),
    )
    qlower = jnp.zeros_like(x.constant)
    qupper = jnp.zeros_like(x.constant)
    for i in range(x.source_size):
        square = _isquare(Interval(dl[i], du[i]))
        term = _iscale(square, 0.5 * x.quadratic[..., i, i])
        qlower, qupper = qlower + term.lower, qupper + term.upper
        for j in range(i + 1, x.source_size):
            monomial = _imul(Interval(dl[i], du[i]), Interval(dl[j], du[j]))
            # Q is symmetric, so the two half-weighted off-diagonal entries
            # combine into this one coefficient.
            coeff = 0.5 * (x.quadratic[..., i, j] + x.quadratic[..., j, i])
            term = _iscale(monomial, coeff)
            qlower, qupper = qlower + term.lower, qupper + term.upper
    return linear, Interval(qlower, qupper)


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
        x.domain_lower,
        x.domain_upper,
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
        like,
    )


def _neg(x):
    if isinstance(x, Interval):
        return _ineg(x)
    if not isinstance(x, TaylorModel):
        # JAX 0.8 Jaxpr literals can be ``TypedNdArray`` instances, which are
        # accepted by primitives but do not implement Python's unary ``-``.
        return lax.neg_p.bind(x)
    return _new(-x.constant, -x.linear, -x.quadratic, _ineg(x.remainder), x)


def _sub(x, y):
    return _add(x, _neg(y))


def _mul(x, y):
    if not isinstance(x, TaylorModel) and not isinstance(y, TaylorModel):
        if isinstance(x, Interval) or isinstance(y, Interval):
            return _imul(_as_interval(x), _as_interval(y))
        return lax.mul_p.bind(x, y)
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
    return _new(c, a, q, remainder, like)


def _is_point_model(x: TaylorModel) -> bool:
    try:
        return bool(
            np.all(np.asarray(x.linear) == 0)
            and np.all(np.asarray(x.quadratic) == 0)
            and np.all(np.asarray(x.remainder.lower) == 0)
            and np.all(np.asarray(x.remainder.upper) == 0)
        )
    except (TypeError, jax.errors.TracerArrayConversionError):
        return False


def _div(x, y):
    if isinstance(y, TaylorModel):
        if _is_point_model(y):
            y = y.constant
        else:
            return _mul(x, _unary_second_order(y, "reciprocal"))
    if isinstance(x, TaylorModel):
        y = jnp.asarray(y)
        return _mul(x, jnp.reciprocal(y))
    if isinstance(x, Interval) or isinstance(y, Interval):
        return _imul(_as_interval(x), _ireciprocal(_as_interval(y)))
    return lax.div_p.bind(x, y)


def _unary_second_order(x: TaylorModel, function: str, exponent=None) -> TaylorModel:
    """Centered order-2 unary rule with an interval Lagrange remainder.

    For ``x = c + l + q + r``, retain

    ``f(c) + f'(c) * (l + q) + 0.5 * f''(c) * l**2``.

    The terms involving ``r``, ``l*q``, ``q**2``, and the third-order
    Lagrange term are enclosed in the returned interval remainder.
    """

    c = x.constant
    input_bounds = taylor_range(x)
    # The Lagrange point lies on the segment between the expansion point c
    # and the represented value. Include c explicitly even for a user-created
    # remainder interval that does not contain zero.
    derivative_lower = jnp.minimum(input_bounds.lower, c)
    derivative_upper = jnp.maximum(input_bounds.upper, c)
    if function == "sin":
        f0, f1, f2 = jnp.sin(c), jnp.cos(c), -jnp.sin(c)
        third = Interval(-jnp.ones_like(c), jnp.ones_like(c))
    elif function == "cos":
        f0, f1, f2 = jnp.cos(c), -jnp.sin(c), -jnp.cos(c)
        third = Interval(-jnp.ones_like(c), jnp.ones_like(c))
    elif function == "sqrt":
        try:
            positive = bool(np.all(np.asarray(derivative_lower) > 0))
        except (TypeError, jax.errors.TracerArrayConversionError):
            positive = False
        if not positive:
            raise NotImplementedError(
                "experimental Taylor sqrt requires a strictly positive range"
            )
        root = jnp.sqrt(c)
        f0, f1, f2 = root, 0.5 / root, -0.25 / (c * root)
        third = Interval(
            3 / (8 * derivative_upper**2.5),
            3 / (8 * derivative_lower**2.5),
        )
    elif function == "power":
        p = jnp.asarray(exponent)
        try:
            valid = p.ndim == 0 and bool(np.all(np.asarray(derivative_lower) > 0))
        except (TypeError, jax.errors.TracerArrayConversionError):
            valid = False
        if not valid:
            raise NotImplementedError(
                "experimental Taylor pow requires a point scalar exponent and "
                "a strictly positive base range"
            )
        f0 = c**p
        f1 = p * c ** (p - 1)
        f2 = p * (p - 1) * c ** (p - 2)
        third_lower_value = p * (p - 1) * (p - 2) * derivative_lower ** (p - 3)
        third_upper_value = p * (p - 1) * (p - 2) * derivative_upper ** (p - 3)
        third = Interval(
            jnp.minimum(third_lower_value, third_upper_value),
            jnp.maximum(third_lower_value, third_upper_value),
        )
    elif function == "reciprocal":
        try:
            excludes_zero = bool(
                np.all(np.asarray((derivative_lower > 0) | (derivative_upper < 0)))
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            excludes_zero = False
        if not excludes_zero:
            raise NotImplementedError(
                "experimental Taylor reciprocal denominator range contains zero: "
                f"lower={np.asarray(derivative_lower)}, "
                f"upper={np.asarray(derivative_upper)}"
            )
        f0 = 1 / c
        f1 = -1 / c**2
        f2 = 2 / c**3
        third_at_lower = -6 / derivative_lower**4
        third_at_upper = -6 / derivative_upper**4
        third = Interval(
            jnp.minimum(third_at_lower, third_at_upper),
            jnp.maximum(third_at_lower, third_at_upper),
        )
    else:  # pragma: no cover - private caller fixes the choices
        raise ValueError(function)
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
    return _new(f0, a, q, remainder, x)


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
        if not _is_point_model(y):
            raise NotImplementedError(
                "experimental Taylor pow requires a point-valued exponent"
            )
        y = y.constant
    if not isinstance(x, TaylorModel):
        return x**y
    try:
        integer = float(np.asarray(y)).is_integer()
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        integer = False
    if integer and int(np.asarray(y)) in (0, 1, 2):
        return _integer_pow(x, y=int(np.asarray(y)))
    return _unary_second_order(x, "power", exponent=y)


def _integer_pow(x, *, y):
    if not isinstance(x, TaylorModel):
        return lax.integer_pow_p.bind(x, y=y)
    if y == 0:
        return constant_taylor_model(
            jnp.ones_like(x.constant), x.domain_lower, x.domain_upper
        )
    if y == 1:
        return x
    if y == 2:
        return _mul(x, x)
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


def _require_fixed_predicate(pred: Interval, operation: str):
    try:
        fixed = np.array_equal(np.asarray(pred.lower), np.asarray(pred.upper))
    except (TypeError, jax.errors.TracerArrayConversionError):
        fixed = False
    if not fixed:
        raise NotImplementedError(
            f"experimental Taylor {operation} requires a sign-fixed predicate"
        )
    return pred.lower


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
        like,
    )


def _select_n(which, *cases):
    if isinstance(which, Interval):
        which = _require_fixed_predicate(which, "select")
    if not any(isinstance(case, TaylorModel) for case in cases):
        if any(isinstance(case, Interval) for case in cases):
            intervals = [
                case if isinstance(case, Interval) else Interval(case, case)
                for case in cases
            ]
            return Interval(
                lax.select_n(which, *[case.lower for case in intervals]),
                lax.select_n(which, *[case.upper for case in intervals]),
            )
        return lax.select_n(which, *cases)
    if len(cases) == 2 and jnp.asarray(which).dtype == jnp.bool_:
        return _where_model(which, cases[1], cases[0])
    like = _template(*cases)
    result = _promote(cases[0], like)
    for index, case in enumerate(cases[1:], start=1):
        result = _where_model(which == index, _promote(case, like), result)
    return result


def _max_or_min(x, y, *, is_max):
    # Promote an Interval to a Taylor remainder when the other operand is a
    # Taylor model.  Handling Interval first would attempt to store a
    # TaylorModel object as an Interval endpoint.
    if isinstance(x, TaylorModel) or isinstance(y, TaylorModel):
        pred = _comparison("ge" if is_max else "le", x, y)
        mask = _require_fixed_predicate(pred, "maximum/minimum")
        like = _template(x, y)
        return _where_model(mask, _promote(x, like), _promote(y, like))
    if isinstance(x, Interval) or isinstance(y, Interval):
        x = x if isinstance(x, Interval) else Interval(x, x)
        y = y if isinstance(y, Interval) else Interval(y, y)
        operation = jnp.maximum if is_max else jnp.minimum
        return Interval(operation(x.lower, y.lower), operation(x.upper, y.upper))
    return jnp.maximum(x, y) if is_max else jnp.minimum(x, y)


def _abs(x):
    if not isinstance(x, TaylorModel):
        return jnp.abs(x)
    bounds = taylor_range(x)
    positive = bounds.lower >= 0
    negative = bounds.upper <= 0
    try:
        fixed = bool(np.all(np.asarray(positive | negative)))
    except (TypeError, jax.errors.TracerArrayConversionError):
        fixed = False
    if not fixed:
        raise NotImplementedError("experimental Taylor abs crosses zero")
    return _where_model(positive, x, -x)


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
        x,
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
        x,
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
        like,
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
        like,
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
        x,
    )


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
