"""Experimental paired-quadratic source and tube operations.

These helpers support the scalar correction in the crossing-hinge diagnostic.
They preserve zero-remainder degree-2 endpoint polynomials on normalized unit
source boxes. They are not a production implicit-certification API.
"""

from __future__ import annotations

import jax.numpy as jnp

from .interval import Interval
from .paired_quadratic import (
    PairedQuadratic, _finish_native, _native_input_valid, _reference_hull,
    pair_status,
)
from .taylor import TaylorModel, append_taylor_sources, polynomial_range


def append_pair_sources(value: PairedQuadratic, count: int) -> PairedQuadratic:
    """Append zero-influence sources after the pair's current source axes."""
    return PairedQuadratic(
        append_taylor_sources(value.lower, count),
        append_taylor_sources(value.upper, count), value.valid,
    )


def singleton_pair(polynomial: TaylorModel) -> PairedQuadratic:
    """Embed one exact quadratic polynomial as a zero-width pair."""
    zero_remainder = ((polynomial.remainder.lower == 0)
                      & (polynomial.remainder.upper == 0))
    return PairedQuadratic(polynomial, polynomial, zero_remainder)


def _correction_midpoint_width(value: PairedQuadratic):
    if value.lower.constant.size != 1:
        raise ValueError("experimental correction lift requires one scalar component")
    return 0.5 * (value.lower + value.upper), 0.5 * (value.upper - value.lower)


def _last_unit_source(value: TaylorModel) -> TaylorModel:
    n = value.source_size + 1
    zeros = jnp.zeros_like(value.constant)
    linear = jnp.zeros(value.shape + (n,), dtype=value.dtype)
    linear = linear.at[..., -1].set(1.0)
    quadratic = jnp.zeros(value.shape + (n, n), dtype=value.dtype)
    return TaylorModel(zeros, linear, quadratic, Interval(zeros, zeros))


def constant_width_correction_lift(value: PairedQuadratic):
    """Singleton M(xi)+rho*eta, with rho >= sup H(xi)."""
    midpoint, half_width = _correction_midpoint_width(value)
    status = pair_status(value)
    radius = polynomial_range(half_width).upper
    model = (append_taylor_sources(midpoint, 1)
             + _last_unit_source(midpoint) * radius)
    valid = (_native_input_valid(value) & (radius >= 0)
             & jnp.isfinite(radius) & status["ordered"])
    return PairedQuadratic(model, model, valid), radius


def affine_width_correction_lift(value: PairedQuadratic):
    """Singleton M(xi)+eta*(c+a.xi+q_upper), exactly degree two."""
    midpoint, half_width = _correction_midpoint_width(value)
    zeros = jnp.zeros_like(half_width.constant)
    quadratic_part = TaylorModel(
        zeros, jnp.zeros_like(half_width.linear), half_width.quadratic,
        Interval(zeros, zeros),
    )
    quadratic_upper = polynomial_range(quadratic_part).upper
    affine_width = TaylorModel(
        half_width.constant + quadratic_upper,
        half_width.linear,
        jnp.zeros_like(half_width.quadratic),
        Interval(zeros, zeros),
    )
    model = (append_taylor_sources(midpoint, 1)
             + append_taylor_sources(affine_width, 1)
             * _last_unit_source(midpoint))
    status = pair_status(value)
    valid = (_native_input_valid(value) & status["ordered"]
             & jnp.isfinite(quadratic_upper))
    return PairedQuadratic(model, model, valid), affine_width, quadratic_upper


def _split_trailing_terms(polynomial: TaylorModel, keep_source_count: int):
    """Keep xi-only coefficients and isolate every eta-containing term."""
    if not 0 <= keep_source_count <= polynomial.source_size:
        raise ValueError("kept source count is outside the polynomial source axes")
    zeros = jnp.zeros_like(polynomial.constant)
    retained = TaylorModel(
        polynomial.constant,
        polynomial.linear[..., :keep_source_count],
        polynomial.quadratic[..., :keep_source_count, :keep_source_count],
        Interval(zeros, zeros),
    )
    trailing_linear = polynomial.linear.at[..., :keep_source_count].set(0)
    trailing_quadratic = polynomial.quadratic.at[
        ..., :keep_source_count, :keep_source_count
    ].set(0)
    trailing = TaylorModel(
        zeros, trailing_linear, trailing_quadratic, Interval(zeros, zeros),
    )
    return retained, trailing


def project_pair_sources(value: PairedQuadratic, keep_source_count: int):
    """Eliminate trailing sources by certified ranges of their polynomial terms."""
    lower_kept, lower_tail = _split_trailing_terms(value.lower, keep_source_count)
    upper_kept, upper_tail = _split_trailing_terms(value.upper, keep_source_count)
    lower_tail_range = polynomial_range(lower_tail)
    upper_tail_range = polynomial_range(upper_tail)
    result = _finish_native(
        lower_kept + lower_tail_range.lower,
        upper_kept + upper_tail_range.upper,
        _native_input_valid(value),
    )
    return result, lower_tail_range, upper_tail_range


def pair_inclusion_margins(outer: PairedQuadratic, inner: PairedQuadratic):
    """Certified lower/upper margins for pointwise inclusion on shared xi."""
    if outer.source_size != inner.source_size:
        raise ValueError("paired inclusion requires the same source count")
    lower = polynomial_range(inner.lower - outer.lower).lower
    upper = polynomial_range(outer.upper - inner.upper).lower
    valid = _native_input_valid(outer, inner)
    return jnp.where(valid, lower, jnp.nan), jnp.where(valid, upper, jnp.nan)


def pair_hull(first: PairedQuadratic, second: PairedQuadratic):
    """Reference-shift quadratic hull of two parameter-only pairs."""
    if first.source_size != second.source_size:
        raise ValueError("paired hull requires the same source count")
    lower, upper = _reference_hull(
        (first.lower, second.lower), (first.upper, second.upper),
    )
    return _finish_native(lower, upper, _native_input_valid(first, second))


def inflate_pair(value: PairedQuadratic, absolute=1e-9):
    """Add one outward constant shift to each polynomial endpoint."""
    absolute = jnp.asarray(absolute)
    valid = _native_input_valid(value) & jnp.isfinite(absolute) & (absolute >= 0)
    return _finish_native(value.lower - absolute, value.upper + absolute, valid)
