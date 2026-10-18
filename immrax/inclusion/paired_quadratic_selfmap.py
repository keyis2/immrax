"""Paired-quadratic source and tube operations for implicit self-maps.

The endpoint polynomials use normalized sources on ``[-1, 1]``.  These
operations preserve the shared source dependence needed to form, project, and
update a paired-quadratic self-map tube.  They do not decide a self-map or
install any inclusion rules.
"""

from __future__ import annotations

import operator

import jax.numpy as jnp

from .interval import Interval
from .paired_quadratic import (
    PairedQuadratic,
    _finish_native,
    _native_input_valid,
    _reference_hull,
)
from .taylor import TaylorModel, append_taylor_sources, polynomial_range


def append_pair_sources(value: PairedQuadratic, count: int) -> PairedQuadratic:
    """Append ``count`` zero-influence sources to both endpoint models."""

    return PairedQuadratic(
        append_taylor_sources(value.lower, count),
        append_taylor_sources(value.upper, count),
        value.valid,
    )


def singleton_pair(polynomial: TaylorModel) -> PairedQuadratic:
    """Embed one exact quadratic polynomial as a zero-width pair."""

    zero_remainder = (
        (polynomial.remainder.lower == 0)
        & (polynomial.remainder.upper == 0)
    )
    return PairedQuadratic(polynomial, polynomial, zero_remainder)


def _outward_ulps(value, count: int):
    count = operator.index(count)
    if count < 0:
        raise ValueError("outward_ulp_steps must be nonnegative")
    result = jnp.asarray(value)
    infinity = jnp.asarray(jnp.inf, dtype=result.dtype)
    for _ in range(count):
        result = jnp.where(
            result != 0,
            jnp.nextafter(result, infinity),
            result,
        )
    return result


def shared_correction_lift(
    value: PairedQuadratic,
    retained_source_count: int,
    *,
    outward_ulp_steps: int = 2,
) -> tuple[PairedQuadratic, jnp.ndarray]:
    """Lift a vector tube into shared correction sources.

    For the floating-point midpoint ``m = (L + U) / 2``, the returned
    singleton represents ``m(xi) + diag(rho) eta``.  One ``eta`` source is
    appended per output coordinate, and

    ``rho_i >= max(sup(m_i - L_i), sup(U_i - m_i))``.

    Thus projecting the correction sources contains the original endpoint
    pair, including rounding introduced while forming ``m``.  The candidate
    must depend only on the retained source coordinates.
    """

    retained_source_count = operator.index(retained_source_count)
    if value.ndim != 1:
        raise ValueError("shared correction lift requires a vector pair")
    if value.source_size != retained_source_count:
        raise ValueError(
            "retained source count must equal the candidate source count; "
            f"got {retained_source_count} for {value.source_size}"
        )

    midpoint = 0.5 * (value.lower + value.upper)
    radius = jnp.maximum(
        polynomial_range(midpoint - value.lower).upper,
        polynomial_range(value.upper - midpoint).upper,
    )
    radius = _outward_ulps(radius, outward_ulp_steps)

    output_size = value.shape[0]
    model = append_taylor_sources(midpoint, output_size)
    rows = jnp.arange(output_size)
    linear = model.linear.at[
        rows, retained_source_count + rows
    ].add(radius)
    zeros = jnp.zeros_like(model.constant)
    model = TaylorModel(
        model.constant,
        linear,
        model.quadratic,
        Interval(zeros, zeros),
    )
    valid = _native_input_valid(value) & jnp.isfinite(radius) & (radius >= 0)
    return PairedQuadratic(model, model, valid), radius


def _split_trailing_terms(polynomial: TaylorModel, keep_source_count: int):
    """Keep retained-source coefficients and isolate every trailing term."""

    keep_source_count = operator.index(keep_source_count)
    if not 0 <= keep_source_count <= polynomial.source_size:
        raise ValueError(
            "kept source count must lie in [0, source_size]; got "
            f"{keep_source_count} for {polynomial.source_size}"
        )
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
        zeros,
        trailing_linear,
        trailing_quadratic,
        Interval(zeros, zeros),
    )
    return retained, trailing


def project_pair_sources(value: PairedQuadratic, keep_source_count: int):
    """Eliminate trailing sources by ranging their polynomial terms."""

    lower_kept, lower_tail = _split_trailing_terms(
        value.lower, keep_source_count
    )
    upper_kept, upper_tail = _split_trailing_terms(
        value.upper, keep_source_count
    )
    lower_tail_range = polynomial_range(lower_tail)
    upper_tail_range = polynomial_range(upper_tail)
    result = _finish_native(
        lower_kept + lower_tail_range.lower,
        upper_kept + upper_tail_range.upper,
        _native_input_valid(value),
    )
    return result, lower_tail_range, upper_tail_range


def pair_inclusion_margins(
    outer: PairedQuadratic, inner: PairedQuadratic
):
    """Return certified lower and upper margins for ``inner`` in ``outer``."""

    if outer.source_size != inner.source_size:
        raise ValueError("paired inclusion requires the same source count")
    lower = polynomial_range(inner.lower - outer.lower).lower
    upper = polynomial_range(outer.upper - inner.upper).lower
    valid = _native_input_valid(outer, inner)
    return jnp.where(valid, lower, jnp.nan), jnp.where(valid, upper, jnp.nan)


def pair_hull(
    first: PairedQuadratic, second: PairedQuadratic
) -> PairedQuadratic:
    """Return a reference-shift quadratic hull of two shared-source pairs."""

    if first.source_size != second.source_size:
        raise ValueError("paired hull requires the same source count")
    lower, upper = _reference_hull(
        (first.lower, second.lower),
        (first.upper, second.upper),
    )
    return _finish_native(lower, upper, _native_input_valid(first, second))


def inflate_pair(value: PairedQuadratic, absolute=1e-9) -> PairedQuadratic:
    """Shift both endpoints outward by one nonnegative constant amount."""

    absolute = jnp.asarray(absolute)
    valid = _native_input_valid(value) & jnp.isfinite(absolute) & (absolute >= 0)
    return _finish_native(
        value.lower - absolute,
        value.upper + absolute,
        valid,
    )


__all__ = [
    "append_pair_sources",
    "inflate_pair",
    "pair_hull",
    "pair_inclusion_margins",
    "project_pair_sources",
    "shared_correction_lift",
    "singleton_pair",
]
