"""Experimental adaptive zonotope bounds for products of exact source planes.

This is deliberately separate from ``AffineBound.product_of_source_planes``.
It searches affine slopes for the exact product of two affine source planes;
it does not select against the production interval or McCormick modes. The
implementation has a fixed nine-candidate JAX graph and uses only array
operations, so it is compatible with ``jit`` and batched output planes.
"""

from __future__ import annotations

import jax.numpy as jnp

from .affine import AffineBound, _bilinear_zonotope_edges

__all__ = ["adaptive_zonotope_source_plane_product"]


def _remainder_oracle(edges, slopes):
    """Extrema and locations of uv-Au-Bv for batched slope candidates."""
    starts, steps, radius = edges
    u0, v0 = starts[..., 0], starts[..., 1]
    du, dv = steps[..., 0], steps[..., 1]
    slope_u, slope_v = slopes[..., 0], slopes[..., 1]

    r0 = (
        u0[..., None, :] * v0[..., None, :]
        - slope_u[..., None] * u0[..., None, :]
        - slope_v[..., None] * v0[..., None, :]
    )
    r1 = (
        u0[..., None, :] * dv[..., None, :]
        + v0[..., None, :] * du[..., None, :]
        - slope_u[..., None] * du[..., None, :]
        - slope_v[..., None] * dv[..., None, :]
    )
    r2 = du * dv
    r2 = r2[..., None, :]
    stationary_t = -r1 / jnp.where(r2 != 0.0, 2.0 * r2, 1.0)
    stationary_valid = (
        (r2 != 0.0) & (stationary_t >= 0.0) & (stationary_t <= 1.0)
    )
    times = jnp.stack(
        (jnp.zeros_like(r0), jnp.ones_like(r0), jnp.clip(stationary_t, 0.0, 1.0)),
        axis=-1,
    )
    values = r0[..., None] + times * (r1[..., None] + times * r2[..., None])
    valid = jnp.stack(
        (jnp.ones_like(stationary_valid), jnp.ones_like(stationary_valid),
         stationary_valid),
        axis=-1,
    )
    points = starts[..., None, :, None, :] + (
        times[..., None] * steps[..., None, :, None, :]
    )
    candidate_count = values.shape[-2] * values.shape[-1]
    values = values.reshape(values.shape[:-2] + (candidate_count,))
    valid = valid.reshape(valid.shape[:-2] + (candidate_count,))
    points = points.reshape(points.shape[:-3] + (candidate_count, 2))
    lower_values = jnp.where(valid, values, jnp.inf)
    upper_values = jnp.where(valid, values, -jnp.inf)
    min_index = jnp.argmin(lower_values, axis=-1)
    max_index = jnp.argmax(upper_values, axis=-1)
    minimum = jnp.take_along_axis(lower_values, min_index[..., None], axis=-1)[..., 0]
    maximum = jnp.take_along_axis(upper_values, max_index[..., None], axis=-1)[..., 0]
    min_point = jnp.take_along_axis(
        points, min_index[..., None, None], axis=-2
    )[..., 0, :]
    max_point = jnp.take_along_axis(
        points, max_index[..., None, None], axis=-2
    )[..., 0, :]

    # The only possible interior stationary point is (u,v)=(B,A).
    stationary_point = jnp.stack((slope_v, slope_u), axis=-1)
    cross = (
        du[..., None, :] * (slope_u[..., :, None] - v0[..., None, :])
        - dv[..., None, :] * (slope_v[..., :, None] - u0[..., None, :])
    )
    tolerance = 64.0 * jnp.finfo(slopes.dtype).eps * (
        1.0 + jnp.max(jnp.abs(starts), axis=(-2, -1))
        + jnp.max(jnp.abs(steps), axis=(-2, -1))
        + jnp.max(jnp.abs(slopes), axis=(-2, -1))
    )
    inside = jnp.all(cross >= -tolerance[..., None, None], axis=-1)
    inside &= (
        (jnp.abs(slope_v) <= radius[..., None, 0] + tolerance[..., None])
        & (jnp.abs(slope_u) <= radius[..., None, 1] + tolerance[..., None])
    )
    stationary_value = -slope_u * slope_v
    use_stationary_min = inside & (stationary_value < minimum)
    use_stationary_max = inside & (stationary_value > maximum)
    minimum = jnp.where(use_stationary_min, stationary_value, minimum)
    maximum = jnp.where(use_stationary_max, stationary_value, maximum)
    min_point = jnp.where(use_stationary_min[..., None], stationary_point, min_point)
    max_point = jnp.where(use_stationary_max[..., None], stationary_point, max_point)

    # The source-box center gives (u,v)=(0,0), hence remainder zero.
    min_point = jnp.where((minimum > 0.0)[..., None], 0.0, min_point)
    max_point = jnp.where((maximum < 0.0)[..., None], 0.0, max_point)
    return jnp.minimum(minimum, 0.0), jnp.maximum(maximum, 0.0), min_point, max_point


def adaptive_zonotope_source_plane_product(
    template: AffineBound,
    first_coeff,
    first_bias,
    second_coeff,
    second_bias,
) -> AffineBound:
    """Bound two exact source planes using nine adaptive zonotope slopes.

    With p=p_c+u and q=q_c+v, each candidate uses the exact decomposition
    pq=p_c q_c+q_c u+p_c v+A u+B v+(uv-Au-Bv). The remainder is optimized
    jointly over the (u,v) zonotope, including edge and interior extrema.

    The center slope is retained among the candidates. Two rounds add four
    subgradient trial slopes each; the complete lower and upper planes are
    selected independently by their extrema over ``template``'s source box.
    """
    first_coeff = jnp.asarray(first_coeff)
    first_bias = jnp.asarray(first_bias)
    second_coeff = jnp.asarray(second_coeff)
    second_bias = jnp.asarray(second_bias)
    source_size = template.input_size
    if first_coeff.ndim == 0 or first_coeff.shape[-1] != source_size:
        raise ValueError("first_coeff must end with template.input_size")
    if second_coeff.ndim == 0 or second_coeff.shape[-1] != source_size:
        raise ValueError("second_coeff must end with template.input_size")
    output_shape = jnp.broadcast_shapes(
        first_coeff.shape[:-1], first_bias.shape,
        second_coeff.shape[:-1], second_bias.shape,
    )
    first_coeff = jnp.broadcast_to(first_coeff, output_shape + (source_size,))
    second_coeff = jnp.broadcast_to(second_coeff, output_shape + (source_size,))
    first_bias = jnp.broadcast_to(first_bias, output_shape)
    second_bias = jnp.broadcast_to(second_bias, output_shape)
    if source_size == 0:
        product = first_bias * second_bias
        zero_coeff = jnp.zeros(output_shape + (0,), dtype=product.dtype)
        return AffineBound(
            zero_coeff, product, zero_coeff, product,
            template.domain_lower, template.domain_upper,
        )

    source_center = 0.5 * (template.domain_lower + template.domain_upper)
    source_radius = 0.5 * (template.domain_upper - template.domain_lower)
    first_center = first_bias + jnp.sum(first_coeff * source_center, axis=-1)
    second_center = second_bias + jnp.sum(second_coeff * source_center, axis=-1)
    edges = _bilinear_zonotope_edges(first_coeff, second_coeff, source_radius)
    radius = edges[2]
    slopes = jnp.zeros(output_shape + (1, 2), dtype=first_center.dtype)
    minima, maxima, min_points, max_points = _remainder_oracle(edges, slopes)
    safe_radius = jnp.where(radius > 0.0, radius, 1.0)
    multipliers = jnp.asarray((0.25, 0.5, 1.0, 2.0), dtype=slopes.dtype)
    multipliers = multipliers.reshape((1,) * len(output_shape) + (4, 1))
    for _ in range(2):
        best_index = jnp.argmin(maxima - minima, axis=-1)
        best_slope = jnp.take_along_axis(
            slopes, best_index[..., None, None], axis=-2
        )[..., 0, :]
        best_min = jnp.take_along_axis(
            minima, best_index[..., None], axis=-1
        )[..., 0]
        best_max = jnp.take_along_axis(
            maxima, best_index[..., None], axis=-1
        )[..., 0]
        min_point = jnp.take_along_axis(
            min_points, best_index[..., None, None], axis=-2
        )[..., 0, :]
        max_point = jnp.take_along_axis(
            max_points, best_index[..., None, None], axis=-2
        )[..., 0, :]
        gradient = (min_point - max_point) / safe_radius
        step = (best_max - best_min) / (jnp.sum(gradient**2, axis=-1) + 1e-24)
        trials = (
            best_slope[..., None, :] * safe_radius[..., None, :]
            - multipliers * step[..., None, None] * gradient[..., None, :]
        ) / safe_radius[..., None, :]
        trial_min, trial_max, trial_min_points, trial_max_points = (
            _remainder_oracle(edges, trials)
        )
        slopes = jnp.concatenate((slopes, trials), axis=-2)
        minima = jnp.concatenate((minima, trial_min), axis=-1)
        maxima = jnp.concatenate((maxima, trial_max), axis=-1)
        min_points = jnp.concatenate((min_points, trial_min_points), axis=-2)
        max_points = jnp.concatenate((max_points, trial_max_points), axis=-2)

    coeff = (
        (second_center[..., None] + slopes[..., 0])[..., None]
        * first_coeff[..., None, :]
        + (first_center[..., None] + slopes[..., 1])[..., None]
        * second_coeff[..., None, :]
    )
    bias = first_center[..., None] * second_center[..., None] - jnp.sum(
        coeff * source_center, axis=-1
    )
    lower_bias = bias + minima
    upper_bias = bias + maxima
    lower_endpoint, _ = template.plane_extrema(coeff, lower_bias)
    _, upper_endpoint = template.plane_extrema(coeff, upper_bias)
    lower_index = jnp.argmax(lower_endpoint, axis=-1)
    upper_index = jnp.argmin(upper_endpoint, axis=-1)
    lower_coeff = jnp.take_along_axis(
        coeff, lower_index[..., None, None], axis=-2
    )[..., 0, :]
    upper_coeff = jnp.take_along_axis(
        coeff, upper_index[..., None, None], axis=-2
    )[..., 0, :]
    lower_bias = jnp.take_along_axis(
        lower_bias, lower_index[..., None], axis=-1
    )[..., 0]
    upper_bias = jnp.take_along_axis(
        upper_bias, upper_index[..., None], axis=-1
    )[..., 0]
    return AffineBound(
        lower_coeff, lower_bias, upper_coeff, upper_bias,
        template.domain_lower, template.domain_upper,
    )
