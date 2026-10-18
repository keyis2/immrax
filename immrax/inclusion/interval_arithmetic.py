"""Explicit outward-expanded operations for :class:`Interval`.

Immrax's natural inclusion transform uses ordinary floating-point endpoint
operations.  Certification pipelines that want a one-representable-value
bookkeeping margin can call the helpers in this module explicitly.  This is
still not a backend-audited directed-rounding implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .interval import Interval, interval
from .jacobian import jacif
from .nif import natif

Array = jax.Array


def _down(value: Array) -> Array:
    value = jnp.asarray(value)
    return jnp.nextafter(value, jnp.full_like(value, -jnp.inf))


def _up(value: Array) -> Array:
    value = jnp.asarray(value)
    return jnp.nextafter(value, jnp.full_like(value, jnp.inf))


def as_interval(value: Interval | Array | float) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


def outward_neg(value: Interval) -> Interval:
    return Interval(_down(-value.upper), _up(-value.lower))


def outward_add(
    first: Interval | Array | float, second: Interval | Array | float
) -> Interval:
    first, second = as_interval(first), as_interval(second)
    return Interval(
        _down(first.lower + second.lower),
        _up(first.upper + second.upper),
    )


def outward_sub(
    first: Interval | Array | float, second: Interval | Array | float
) -> Interval:
    first, second = as_interval(first), as_interval(second)
    return Interval(
        _down(first.lower - second.upper),
        _up(first.upper - second.lower),
    )


def outward_mul(
    first: Interval | Array | float, second: Interval | Array | float
) -> Interval:
    first, second = as_interval(first), as_interval(second)
    products = jnp.stack(
        (
            first.lower * second.lower,
            first.lower * second.upper,
            first.upper * second.lower,
            first.upper * second.upper,
        )
    )
    return Interval(_down(jnp.min(products, axis=0)), _up(jnp.max(products, axis=0)))


def outward_div(
    numerator: Interval | Array | float,
    denominator: Interval | Array | float,
) -> Interval:
    denominator = as_interval(denominator)
    reciprocal = Interval(
        _down(1.0 / denominator.upper),
        _up(1.0 / denominator.lower),
    )
    return outward_mul(numerator, reciprocal)


def interval_matmul(a: Interval | Array, b: Interval | Array) -> Interval:
    """Outward-expanded interval matrix product with broadcast batch axes."""

    a, b = as_interval(a), as_interval(b)
    if a.lower.ndim < 2 or b.lower.ndim < 2:
        raise ValueError("interval_matmul expects arrays with at least two dimensions")
    if a.lower.shape[-1] != b.lower.shape[-2]:
        raise ValueError(f"incompatible matrix shapes {a.lower.shape} and {b.lower.shape}")

    batch_shape = jnp.broadcast_shapes(a.lower.shape[:-2], b.lower.shape[:-2])
    m, k_dim = a.lower.shape[-2:]
    n = b.lower.shape[-1]
    a_lower = jnp.broadcast_to(a.lower, batch_shape + (m, k_dim))
    a_upper = jnp.broadcast_to(a.upper, batch_shape + (m, k_dim))
    b_lower = jnp.broadcast_to(b.lower, batch_shape + (k_dim, n))
    b_upper = jnp.broadcast_to(b.upper, batch_shape + (k_dim, n))
    zeros = jnp.zeros(batch_shape + (m, n), dtype=jnp.result_type(a.lower, b.lower))

    def accumulate(k, bounds):
        lower, upper = bounds
        al = a_lower[..., :, k, None]
        au = a_upper[..., :, k, None]
        bl = b_lower[..., None, k, :]
        bu = b_upper[..., None, k, :]
        products = jnp.stack((al * bl, al * bu, au * bl, au * bu))
        term_lower = _down(jnp.min(products, axis=0))
        term_upper = _up(jnp.max(products, axis=0))
        return _down(lower + term_lower), _up(upper + term_upper)

    lower, upper = jax.lax.fori_loop(0, k_dim, accumulate, (zeros, zeros))
    return Interval(lower, upper)


def interval_matvec(a: Interval | Array, x: Interval | Array) -> Interval:
    """Outward-expanded interval matrix-vector product."""

    a, x = as_interval(a), as_interval(x)
    if a.lower.ndim < 2 or x.lower.ndim < 1:
        raise ValueError("interval_matvec expects a matrix and a vector")
    result = interval_matmul(a, Interval(x.lower[..., :, None], x.upper[..., :, None]))
    return Interval(result.lower[..., 0], result.upper[..., 0])


def affine_image(offset: Array, matrix: Array, delta: Interval) -> Interval:
    return outward_add(Interval.point(offset), interval_matvec(matrix, delta))


def scale_interval(value: Interval, scale: Array) -> Interval:
    """Scale an interval by positive componentwise factors."""

    scale = jnp.asarray(scale)
    return Interval(_down(value.lower * scale), _up(value.upper * scale))


def scale_jacobian_interval(
    jacobian: Interval, row_scale: Array, column_scale: Array
) -> Interval:
    """Apply positive diagonal row and column scaling to an interval Jacobian."""

    return outward_mul(jacobian, row_scale[:, None] * column_scale[None, :])


def interval_tensor_lastdim_matvec(
    tensor: Interval, vector: Interval
) -> Interval:
    """Contract an interval tensor and vector over their last axis."""

    tensor_matrix = Interval(
        tensor.lower.reshape((-1, tensor.lower.shape[-1])),
        tensor.upper.reshape((-1, tensor.upper.shape[-1])),
    )
    product = interval_matvec(tensor_matrix, vector)
    return Interval(
        product.lower.reshape(tensor.lower.shape[:-1]),
        product.upper.reshape(tensor.upper.shape[:-1]),
    )


def interval_function(function, *boxes: Interval) -> Interval:
    """Evaluate a JAX function over interval arguments using ``natif``."""

    result = natif(function)(*[interval(box.lower, box.upper) for box in boxes])
    if not isinstance(result, Interval):
        return Interval.point(jnp.asarray(result))
    return result


def interval_function_jacif(function, *boxes: Interval) -> Interval:
    """Evaluate a JAX function over interval arguments using ``jacif``."""

    result = jacif(function)(*[interval(box.lower, box.upper) for box in boxes])
    if not isinstance(result, Interval):
        return Interval.point(jnp.asarray(result))
    return result


def _contains_phase(lower: Array, upper: Array, offset: float, period: float) -> Array:
    first = jnp.ceil((lower - offset) / period)
    last = jnp.floor((upper - offset) / period)
    return first <= last


def interval_sin(x: Interval) -> Interval:
    endpoint_lower = jnp.minimum(jnp.sin(x.lower), jnp.sin(x.upper))
    endpoint_upper = jnp.maximum(jnp.sin(x.lower), jnp.sin(x.upper))
    full_period = x.upper - x.lower >= 2.0 * jnp.pi
    has_maximum = _contains_phase(x.lower, x.upper, jnp.pi / 2.0, 2.0 * jnp.pi)
    has_minimum = _contains_phase(x.lower, x.upper, 3.0 * jnp.pi / 2.0, 2.0 * jnp.pi)
    lower = jnp.where(full_period | has_minimum, -1.0, endpoint_lower)
    upper = jnp.where(full_period | has_maximum, 1.0, endpoint_upper)
    return Interval(_down(lower), _up(upper))


def interval_cos(x: Interval) -> Interval:
    endpoint_lower = jnp.minimum(jnp.cos(x.lower), jnp.cos(x.upper))
    endpoint_upper = jnp.maximum(jnp.cos(x.lower), jnp.cos(x.upper))
    full_period = x.upper - x.lower >= 2.0 * jnp.pi
    has_maximum = _contains_phase(x.lower, x.upper, 0.0, 2.0 * jnp.pi)
    has_minimum = _contains_phase(x.lower, x.upper, jnp.pi, 2.0 * jnp.pi)
    lower = jnp.where(full_period | has_minimum, -1.0, endpoint_lower)
    upper = jnp.where(full_period | has_maximum, 1.0, endpoint_upper)
    return Interval(_down(lower), _up(upper))


def interval_sigmoid(x: Interval) -> Interval:
    return Interval(_down(jax.nn.sigmoid(x.lower)), _up(jax.nn.sigmoid(x.upper)))


def interval_softplus(x: Interval) -> Interval:
    return Interval(_down(jax.nn.softplus(x.lower)), _up(jax.nn.softplus(x.upper)))
