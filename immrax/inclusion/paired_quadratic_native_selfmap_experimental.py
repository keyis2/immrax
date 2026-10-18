"""Experimental native source and tube operations for implicit self-maps.

Ports of the supported ``paired_quadratic_selfmap`` helpers to the native
remainder-free ``PairedQuadratic``. They preserve the shared source
dependence needed to lift, project and update a paired-quadratic self-map
tube; they decide nothing and install no rules. Validity is per output
element. Opt-in by full module path; not exported.
"""

from __future__ import annotations

import operator

import jax.numpy as jnp

from .paired_quadratic_native_experimental import (
    PairedQuadratic,
    _finish,
    _input_valid,
    _lower_reference_hull,
    _upper_reference_hull,
)
from .quadratic_polynomial_experimental import QuadraticPolynomial, quadratic_range


def _append_sources(p: QuadraticPolynomial, count: int) -> QuadraticPolynomial:
    padding = ((0, 0),) * p.ndim
    return QuadraticPolynomial(p.constant, jnp.pad(p.linear, padding + ((0, count),)),
                               jnp.pad(p.quadratic, padding + ((0, count), (0, count))))


def append_pair_sources(value: PairedQuadratic, count: int) -> PairedQuadratic:
    """Append ``count`` zero-influence normalized sources, ordered last."""

    count = operator.index(count)
    if count < 0:
        raise ValueError("appended source count must be nonnegative")
    return PairedQuadratic(_append_sources(value.lower, count), _append_sources(value.upper, count), value.valid)


def singleton_pair(polynomial: QuadraticPolynomial) -> PairedQuadratic:
    """Embed one exact quadratic polynomial as a zero-width pair."""

    return PairedQuadratic(polynomial, polynomial)


def _outward_ulps(value, count: int):
    count = operator.index(count)
    if count < 0:
        raise ValueError("outward_ulp_steps must be nonnegative")
    result = jnp.asarray(value)
    infinity = jnp.asarray(jnp.inf, dtype=result.dtype)
    for _ in range(count):
        result = jnp.where(result != 0, jnp.nextafter(result, infinity), result)
    return result


def shared_correction_lift(value: PairedQuadratic, retained_source_count: int, *, outward_ulp_steps: int = 2):
    """Lift a vector tube into one shared correction source per output.

    For the floating-point midpoint ``m = (L + U) / 2`` the singleton
    represents ``m(xi) + diag(rho) eta`` with

    ``rho_i >= max(sup(m_i - L_i), sup(U_i - m_i))``

    (then a fixed outward-ULP step), so projecting ``eta`` contains the
    original pair including the rounding of ``m``. Returns the lift and rho.
    """

    retained_source_count = operator.index(retained_source_count)
    if value.ndim != 1:
        raise ValueError("shared correction lift requires a vector pair")
    if value.source_size != retained_source_count:
        raise ValueError("retained source count must equal the candidate source count; "
                         f"got {retained_source_count} for {value.source_size}")
    midpoint = 0.5 * (value.lower + value.upper)
    radius = jnp.maximum(quadratic_range(midpoint - value.lower).upper,
                         quadratic_range(value.upper - midpoint).upper)
    radius = _outward_ulps(radius, outward_ulp_steps)
    output_size = value.shape[0]
    lifted = _append_sources(midpoint, output_size)
    rows = jnp.arange(output_size)
    lifted = QuadraticPolynomial(lifted.constant, lifted.linear.at[rows, retained_source_count + rows].add(radius),
                                 lifted.quadratic)
    valid = _input_valid(value) & jnp.isfinite(radius) & (radius >= 0)
    return PairedQuadratic(lifted, lifted, valid), radius


def _split_trailing_terms(p: QuadraticPolynomial, keep_source_count: int):
    """Keep retained-source coefficients; isolate every term with a trailing source."""

    keep_source_count = operator.index(keep_source_count)
    if not 0 <= keep_source_count <= p.source_size:
        raise ValueError(f"kept source count must lie in [0, source_size]; got {keep_source_count} for {p.source_size}")
    k = keep_source_count
    retained = QuadraticPolynomial(p.constant, p.linear[..., :k], p.quadratic[..., :k, :k])
    trailing = QuadraticPolynomial(jnp.zeros_like(p.constant), p.linear.at[..., :k].set(0),
                                   p.quadratic.at[..., :k, :k].set(0))
    return retained, trailing


def project_pair_sources(value: PairedQuadratic, keep_source_count: int):
    """Eliminate trailing sources by ranging every term that contains one.

    Returns the projected pair and the lower/upper eliminated-term ranges.
    """

    lower_kept, lower_tail = _split_trailing_terms(value.lower, keep_source_count)
    upper_kept, upper_tail = _split_trailing_terms(value.upper, keep_source_count)
    lower_tail_range, upper_tail_range = quadratic_range(lower_tail), quadratic_range(upper_tail)
    result = _finish(lower_kept + lower_tail_range.lower, upper_kept + upper_tail_range.upper, _input_valid(value))
    return result, lower_tail_range, upper_tail_range


def pair_inclusion_margins(outer: PairedQuadratic, inner: PairedQuadratic):
    """Termwise lower bounds of ``inner.L - outer.L`` and ``outer.U - inner.U``."""

    if outer.source_size != inner.source_size:
        raise ValueError("paired inclusion requires the same source count")
    lower = quadratic_range(inner.lower - outer.lower).lower
    upper = quadratic_range(outer.upper - inner.upper).lower
    valid = _input_valid(outer, inner)
    return jnp.where(valid, lower, jnp.nan), jnp.where(valid, upper, jnp.nan)


def pair_hull(first: PairedQuadratic, second: PairedQuadratic) -> PairedQuadratic:
    """Reference-shift quadratic hull of two shared-source pairs."""

    if first.source_size != second.source_size:
        raise ValueError("paired hull requires the same source count")
    lower = _lower_reference_hull((first.lower, second.lower))
    upper = _upper_reference_hull((first.upper, second.upper))
    return _finish(lower, upper, _input_valid(first, second))


def inflate_pair(value: PairedQuadratic, absolute=1e-9) -> PairedQuadratic:
    """Shift both endpoints outward by one nonnegative constant amount."""

    absolute = jnp.asarray(absolute)
    valid = _input_valid(value) & jnp.isfinite(absolute) & (absolute >= 0)
    return _finish(value.lower - absolute, value.upper + absolute, valid)
