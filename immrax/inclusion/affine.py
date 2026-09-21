"""Affine lower/upper bounds expressed in the original interval inputs."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from .interval import Interval


def _tighter_concretized_planes(first: AffineBound, second: AffineBound) -> AffineBound:
    """Select each complete affine endpoint by its source-box extremum."""
    use_first_lower = first.lower >= second.lower
    use_first_upper = first.upper <= second.upper
    return AffineBound(
        jnp.where(use_first_lower[..., None], first.lower_coeff, second.lower_coeff),
        jnp.where(use_first_lower, first.lower_bias, second.lower_bias),
        jnp.where(use_first_upper[..., None], first.upper_coeff, second.upper_coeff),
        jnp.where(use_first_upper, first.upper_bias, second.upper_bias),
        first.domain_lower,
        first.domain_upper,
    )


def _bilinear_zonotope_edges(first_coeff, second_coeff, source_radius):
    """Return boundary starts and steps for a two-dimensional zonotope."""
    generators = jnp.stack(
        (first_coeff * source_radius, second_coeff * source_radius), axis=-1
    )
    flip = (generators[..., 1] < 0.0) | (
        (generators[..., 1] == 0.0) & (generators[..., 0] < 0.0)
    )
    oriented = jnp.where(flip[..., None], -generators, generators)
    order = jnp.argsort(jnp.arctan2(oriented[..., 1], oriented[..., 0]), axis=-1)
    ordered = jnp.take_along_axis(oriented, order[..., None], axis=-2)
    steps = jnp.concatenate((2.0 * ordered, -2.0 * ordered), axis=-2)
    previous_steps = jnp.concatenate(
        (jnp.zeros_like(steps[..., :1, :]), jnp.cumsum(steps[..., :-1, :], axis=-2)),
        axis=-2,
    )
    edge_start = -jnp.sum(ordered, axis=-2)[..., None, :] + previous_steps
    radius = jnp.sum(jnp.abs(generators), axis=-2)
    return edge_start, steps, radius


def _bilinear_zonotope_extrema(first_coeff, second_coeff, source_radius):
    """Exact extrema of ``(a.T @ xi) * (c.T @ xi)`` on a source box."""
    edge_start, steps, _ = _bilinear_zonotope_edges(
        first_coeff, second_coeff, source_radius
    )
    u0, v0 = edge_start[..., 0], edge_start[..., 1]
    du, dv = steps[..., 0], steps[..., 1]
    r0 = u0 * v0
    r1 = u0 * dv + v0 * du
    r2 = du * dv
    stationary_t = -r1 / jnp.where(r2 != 0.0, 2.0 * r2, 1.0)
    stationary_value = r0 + stationary_t * (r1 + stationary_t * r2)
    interior = (r2 != 0.0) & (stationary_t >= 0.0) & (stationary_t <= 1.0)
    minimum = jnp.min(
        jnp.minimum(
            jnp.minimum(r0, r0 + r1 + r2),
            jnp.where(interior, stationary_value, jnp.inf),
        ),
        axis=-1,
    )
    maximum = jnp.max(
        jnp.maximum(
            jnp.maximum(r0, r0 + r1 + r2),
            jnp.where(interior, stationary_value, -jnp.inf),
        ),
        axis=-1,
    )
    # The centered source box contains xi=0, so (u,v)=(0,0) is feasible.
    return jnp.minimum(minimum, 0.0), jnp.maximum(maximum, 0.0)


def _adaptive_bilinear_remainder_oracle(edges, slopes):
    """Extrema and locations of ``uv - A u - B v`` for slope candidates."""
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
    r2 = (du * dv)[..., None, :]
    stationary_t = -r1 / jnp.where(r2 != 0.0, 2.0 * r2, 1.0)
    stationary_valid = (
        (r2 != 0.0) & (stationary_t >= 0.0) & (stationary_t <= 1.0)
    )
    times = jnp.stack(
        (
            jnp.zeros_like(r0),
            jnp.ones_like(r0),
            jnp.clip(stationary_t, 0.0, 1.0),
        ),
        axis=-1,
    )
    values = r0[..., None] + times * (r1[..., None] + times * r2[..., None])
    valid = jnp.stack(
        (
            jnp.ones_like(stationary_valid),
            jnp.ones_like(stationary_valid),
            stationary_valid,
        ),
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
    minimum = jnp.take_along_axis(
        lower_values, min_index[..., None], axis=-1
    )[..., 0]
    maximum = jnp.take_along_axis(
        upper_values, max_index[..., None], axis=-1
    )[..., 0]
    min_point = jnp.take_along_axis(
        points, min_index[..., None, None], axis=-2
    )[..., 0, :]
    max_point = jnp.take_along_axis(
        points, max_index[..., None, None], axis=-2
    )[..., 0, :]

    # The only possible interior stationary point is (u, v) = (B, A).
    stationary_point = jnp.stack((slope_v, slope_u), axis=-1)
    cross = (
        du[..., None, :] * (slope_u[..., :, None] - v0[..., None, :])
        - dv[..., None, :] * (slope_v[..., :, None] - u0[..., None, :])
    )
    tolerance = 64.0 * jnp.finfo(slopes.dtype).eps * (
        1.0
        + jnp.max(jnp.abs(starts), axis=(-2, -1))
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
    min_point = jnp.where(
        use_stationary_min[..., None], stationary_point, min_point
    )
    max_point = jnp.where(
        use_stationary_max[..., None], stationary_point, max_point
    )

    # The centered source box contains (u, v) = (0, 0).
    min_point = jnp.where((minimum > 0.0)[..., None], 0.0, min_point)
    max_point = jnp.where((maximum < 0.0)[..., None], 0.0, max_point)
    return (
        jnp.minimum(minimum, 0.0),
        jnp.maximum(maximum, 0.0),
        min_point,
        max_point,
    )


def _adaptive_zonotope_source_plane_product(
    template, first_coeff, first_bias, second_coeff, second_bias
):
    """Bound two source planes with a fixed nine-slope zonotope search."""
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
        first_coeff.shape[:-1],
        first_bias.shape,
        second_coeff.shape[:-1],
        second_bias.shape,
    )
    first_coeff = jnp.broadcast_to(first_coeff, output_shape + (source_size,))
    second_coeff = jnp.broadcast_to(second_coeff, output_shape + (source_size,))
    first_bias = jnp.broadcast_to(first_bias, output_shape)
    second_bias = jnp.broadcast_to(second_bias, output_shape)
    if source_size == 0:
        product = first_bias * second_bias
        zero_coeff = jnp.zeros(output_shape + (0,), dtype=product.dtype)
        return AffineBound(
            zero_coeff,
            product,
            zero_coeff,
            product,
            template.domain_lower,
            template.domain_upper,
        )

    source_center = 0.5 * (template.domain_lower + template.domain_upper)
    source_radius = 0.5 * (template.domain_upper - template.domain_lower)
    first_center = first_bias + jnp.sum(first_coeff * source_center, axis=-1)
    second_center = second_bias + jnp.sum(second_coeff * source_center, axis=-1)
    edges = _bilinear_zonotope_edges(first_coeff, second_coeff, source_radius)
    radius = edges[2]
    slopes = jnp.zeros(output_shape + (1, 2), dtype=first_center.dtype)
    minima, maxima, min_points, max_points = (
        _adaptive_bilinear_remainder_oracle(edges, slopes)
    )
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
            _adaptive_bilinear_remainder_oracle(edges, trials)
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
    adaptive = AffineBound(
        lower_coeff,
        lower_bias,
        upper_coeff,
        upper_bias,
        template.domain_lower,
        template.domain_upper,
    )
    interval = template.product_of_source_planes(
        first_coeff,
        first_bias,
        second_coeff,
        second_bias,
        quadratic_relaxation="interval",
    )
    return _tighter_concretized_planes(adaptive, interval)


@register_pytree_node_class
class AffineBound:
    """Componentwise affine lower and upper bounds over one input box.

    For a value with shape ``shape`` and an input box with ``n`` scalar
    coordinates, coefficient arrays have shape ``shape + (n,)`` and bias
    arrays have shape ``shape``.  All values produced during one evaluation
    share ``domain_lower`` and ``domain_upper``.
    """

    def __init__(
        self,
        lower_coeff: ArrayLike,
        lower_bias: ArrayLike,
        upper_coeff: ArrayLike,
        upper_bias: ArrayLike,
        domain_lower: ArrayLike,
        domain_upper: ArrayLike,
    ) -> None:
        self.lower_coeff = jnp.asarray(lower_coeff)
        self.lower_bias = jnp.asarray(lower_bias)
        self.upper_coeff = jnp.asarray(upper_coeff)
        self.upper_bias = jnp.asarray(upper_bias)
        self.domain_lower = jnp.asarray(domain_lower).reshape(-1)
        self.domain_upper = jnp.asarray(domain_upper).reshape(-1)

        expected = self.lower_bias.shape + (self.domain_lower.size,)
        if self.lower_coeff.shape != expected or self.upper_coeff.shape != expected:
            raise ValueError(
                "Affine coefficient shape must be value shape + (input_size,); "
                f"expected {expected}, got {self.lower_coeff.shape} and "
                f"{self.upper_coeff.shape}."
            )
        if self.upper_bias.shape != self.lower_bias.shape:
            raise ValueError("Lower and upper affine biases must have the same shape.")
        if self.domain_lower.shape != self.domain_upper.shape:
            raise ValueError(
                "Input-box lower and upper bounds must have the same shape."
            )

    def tree_flatten(self):
        return (
            (
                self.lower_coeff,
                self.lower_bias,
                self.upper_coeff,
                self.upper_bias,
                self.domain_lower,
                self.domain_upper,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.lower_bias.shape

    @property
    def ndim(self) -> int:
        return self.lower_bias.ndim

    @property
    def size(self) -> int:
        return self.lower_bias.size

    @property
    def input_size(self) -> int:
        return self.domain_lower.size

    @property
    def dtype(self):
        return self.lower_bias.dtype

    @staticmethod
    def _minimum(coeff, bias, domain_lower, domain_upper):
        positive = jnp.maximum(coeff, 0)
        negative = jnp.minimum(coeff, 0)
        return (
            jnp.sum(positive * domain_lower + negative * domain_upper, axis=-1) + bias
        )

    @staticmethod
    def _maximum(coeff, bias, domain_lower, domain_upper):
        positive = jnp.maximum(coeff, 0)
        negative = jnp.minimum(coeff, 0)
        return (
            jnp.sum(positive * domain_upper + negative * domain_lower, axis=-1) + bias
        )

    @staticmethod
    def _interval_product(first_lower, first_upper, second_lower, second_upper):
        products = jnp.stack(
            (
                first_lower * second_lower,
                first_lower * second_upper,
                first_upper * second_lower,
                first_upper * second_upper,
            ),
            axis=0,
        )
        return jnp.min(products, axis=0), jnp.max(products, axis=0)

    @property
    def lower(self):
        """Concretized componentwise lower endpoint."""
        return self._minimum(
            self.lower_coeff, self.lower_bias, self.domain_lower, self.domain_upper
        )

    @property
    def upper(self):
        """Concretized componentwise upper endpoint."""
        return self._maximum(
            self.upper_coeff, self.upper_bias, self.domain_lower, self.domain_upper
        )

    @property
    def center(self):
        return (self.lower + self.upper) / 2

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def pert(self):
        return self.width / 2

    @property
    def T(self):
        return self.transpose()

    # Short mathematical aliases.
    @property
    def A_l(self):
        return self.lower_coeff

    @property
    def b_l(self):
        return self.lower_bias

    @property
    def A_u(self):
        return self.upper_coeff

    @property
    def b_u(self):
        return self.upper_bias

    def concretize(self) -> Interval:
        return Interval(self.lower, self.upper)

    def plane_extrema(self, coeff: ArrayLike, bias: ArrayLike):
        """Return the exact extrema of an affine plane over this source box."""
        coeff = jnp.asarray(coeff)
        bias = jnp.asarray(bias)
        expected = bias.shape + (self.input_size,)
        if coeff.shape != expected:
            raise ValueError(
                "Affine plane coefficient shape must be bias shape + "
                f"(input_size,); expected {expected}, got {coeff.shape}."
            )
        return (
            self._minimum(coeff, bias, self.domain_lower, self.domain_upper),
            self._maximum(coeff, bias, self.domain_lower, self.domain_upper),
        )

    def product_of_source_planes(
        self,
        first_coeff: ArrayLike,
        first_bias: ArrayLike,
        second_coeff: ArrayLike,
        second_bias: ArrayLike,
        *,
        quadratic_relaxation: Literal[
            "affine", "interval", "zonotope", "adaptive", "best"
        ] = "interval",
    ) -> AffineBound:
        """Bound the product of two affine planes in this source domain.

        For ``f(z) = a^T z + b`` and ``g(z) = c^T z + d``, the exact product
        is

        ``bd + (d a + b c)^T z + sum_i a_i c_i z_i^2``
        ``+ sum_{i<j} (a_i c_j + a_j c_i) z_i z_j``.

        The constant and linear terms are retained exactly.  With
        ``quadratic_relaxation="affine"``, each square is bounded by its
        tangent and endpoint secant, while each distinct-source product uses
        McCormick planes.  With ``quadratic_relaxation="interval"``, exact
        interval ranges of the individual quadratic terms are added to the
        biases instead. ``quadratic_relaxation="zonotope"`` optimizes the
        centered bilinear remainder jointly over its two-dimensional source
        zonotope, then selects against ``"interval"`` at each complete
        endpoint. ``"adaptive"`` retains the center-slope zonotope candidate,
        adds eight fixed subgradient trial slopes, and selects against
        ``"interval"``. ``"best"`` selects between ``"adaptive"`` and
        ``"affine"``.
        The mode is a static tracing choice; all paths use vectorized JAX
        operations.
        """
        if quadratic_relaxation == "best":
            affine = self.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="affine",
            )
            adaptive = self.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="adaptive",
            )
            return _tighter_concretized_planes(affine, adaptive)
        if quadratic_relaxation not in (
            "affine", "interval", "zonotope", "adaptive"
        ):
            raise ValueError(
                "quadratic_relaxation must be 'affine', 'interval', "
                "'zonotope', 'adaptive', or 'best'; "
                f"got {quadratic_relaxation!r}."
            )
        if quadratic_relaxation == "adaptive":
            return _adaptive_zonotope_source_plane_product(
                self,
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
            )
        first_coeff = jnp.asarray(first_coeff)
        first_bias = jnp.asarray(first_bias)
        second_coeff = jnp.asarray(second_coeff)
        second_bias = jnp.asarray(second_bias)
        if first_coeff.ndim == 0 or first_coeff.shape[-1] != self.input_size:
            raise ValueError(
                "First affine-plane coefficient shape must end with input_size; "
                f"expected {self.input_size}, got {first_coeff.shape}."
            )
        if second_coeff.ndim == 0 or second_coeff.shape[-1] != self.input_size:
            raise ValueError(
                "Second affine-plane coefficient shape must end with input_size; "
                f"expected {self.input_size}, got {second_coeff.shape}."
            )
        output_shape = jnp.broadcast_shapes(
            first_coeff.shape[:-1],
            first_bias.shape,
            second_coeff.shape[:-1],
            second_bias.shape,
        )
        coeff_shape = output_shape + (self.input_size,)
        first_coeff = jnp.broadcast_to(first_coeff, coeff_shape)
        second_coeff = jnp.broadcast_to(second_coeff, coeff_shape)
        first_bias = jnp.broadcast_to(first_bias, output_shape)
        second_bias = jnp.broadcast_to(second_bias, output_shape)

        # Exact constant and linear terms.
        linear_coeff = (
            second_bias[..., None] * first_coeff + first_bias[..., None] * second_coeff
        )
        constant = first_bias * second_bias

        source_lower = self.domain_lower
        source_upper = self.domain_upper
        source_center = 0.5 * (source_lower + source_upper)

        if quadratic_relaxation == "zonotope":
            # Center the box: p=p_c+a^T xi, q=q_c+c^T xi. Retain the
            # entire first-order part and range (a^T xi)(c^T xi) jointly.
            first_center = first_bias + jnp.sum(first_coeff * source_center, axis=-1)
            second_center = second_bias + jnp.sum(
                second_coeff * source_center, axis=-1
            )
            centered_coeff = (
                second_center[..., None] * first_coeff
                + first_center[..., None] * second_coeff
            )
            centered_bias = first_center * second_center - jnp.sum(
                centered_coeff * source_center, axis=-1
            )
            if self.input_size:
                remainder_lower, remainder_upper = _bilinear_zonotope_extrema(
                    first_coeff,
                    second_coeff,
                    0.5 * (source_upper - source_lower),
                )
            else:
                remainder_lower = jnp.zeros(output_shape, dtype=centered_bias.dtype)
                remainder_upper = remainder_lower
            joint = AffineBound(
                centered_coeff,
                centered_bias + remainder_lower,
                centered_coeff,
                centered_bias + remainder_upper,
                source_lower,
                source_upper,
            )
            # The existing interval mode ranges uncentered monomials, so its
            # final endpoint may be tighter despite the joint remainder.
            interval = self.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="interval",
            )
            return _tighter_concretized_planes(joint, interval)

        if quadratic_relaxation == "interval":
            square_at_lower = source_lower**2
            square_at_upper = source_upper**2
            square_lower = jnp.where(
                (source_lower <= 0.0) & (source_upper >= 0.0),
                0.0,
                jnp.minimum(square_at_lower, square_at_upper),
            )
            square_upper = jnp.maximum(square_at_lower, square_at_upper)
            diagonal_coefficient = first_coeff * second_coeff
            diagonal_positive = diagonal_coefficient >= 0.0
            quadratic_lower = jnp.sum(
                diagonal_coefficient
                * jnp.where(diagonal_positive, square_lower, square_upper),
                axis=-1,
            )
            quadratic_upper = jnp.sum(
                diagonal_coefficient
                * jnp.where(diagonal_positive, square_upper, square_lower),
                axis=-1,
            )

            first_index, second_index = jnp.triu_indices(self.input_size, k=1)
            lower_i = source_lower[first_index]
            upper_i = source_upper[first_index]
            lower_j = source_lower[second_index]
            upper_j = source_upper[second_index]
            cross_coefficient = (
                first_coeff[..., first_index] * second_coeff[..., second_index]
                + first_coeff[..., second_index] * second_coeff[..., first_index]
            )
            cross_products = jnp.stack(
                (
                    lower_i * lower_j,
                    lower_i * upper_j,
                    upper_i * lower_j,
                    upper_i * upper_j,
                ),
                axis=0,
            )
            cross_lower = jnp.min(cross_products, axis=0)
            cross_upper = jnp.max(cross_products, axis=0)
            cross_positive = cross_coefficient >= 0.0
            quadratic_lower = quadratic_lower + jnp.sum(
                cross_coefficient * jnp.where(cross_positive, cross_lower, cross_upper),
                axis=-1,
            )
            quadratic_upper = quadratic_upper + jnp.sum(
                cross_coefficient * jnp.where(cross_positive, cross_upper, cross_lower),
                axis=-1,
            )
            return AffineBound(
                linear_coeff,
                constant + quadratic_lower,
                linear_coeff,
                constant + quadratic_upper,
                source_lower,
                source_upper,
            )

        # For z_i^2, use the tangent at the point in [l_i, u_i] nearest zero
        # as the lower plane and the endpoint secant as the upper plane.
        tangent_point = jnp.clip(
            jnp.zeros_like(source_lower), source_lower, source_upper
        )
        square_lower_slope = 2.0 * tangent_point
        square_lower_bias = -(tangent_point**2)
        square_upper_slope = source_lower + source_upper
        square_upper_bias = -source_lower * source_upper

        diagonal_coefficient = first_coeff * second_coeff
        diagonal_positive = diagonal_coefficient >= 0.0
        lower_coeff = linear_coeff + diagonal_coefficient * jnp.where(
            diagonal_positive, square_lower_slope, square_upper_slope
        )
        lower_bias = constant + jnp.sum(
            diagonal_coefficient
            * jnp.where(diagonal_positive, square_lower_bias, square_upper_bias),
            axis=-1,
        )
        upper_coeff = linear_coeff + diagonal_coefficient * jnp.where(
            diagonal_positive, square_upper_slope, square_lower_slope
        )
        upper_bias = constant + jnp.sum(
            diagonal_coefficient
            * jnp.where(diagonal_positive, square_upper_bias, square_lower_bias),
            axis=-1,
        )

        # Vectorize the i < j terms.  For z_i z_j, L1/L2 and U1/U2 are the
        # two McCormick lower and upper planes.  A negative quadratic
        # coefficient exchanges lower and upper candidates.
        first_index, second_index = jnp.triu_indices(self.input_size, k=1)
        lower_i = source_lower[first_index]
        upper_i = source_upper[first_index]
        lower_j = source_lower[second_index]
        upper_j = source_upper[second_index]
        center_i = source_center[first_index]
        center_j = source_center[second_index]

        cross_coefficient = (
            first_coeff[..., first_index] * second_coeff[..., second_index]
            + first_coeff[..., second_index] * second_coeff[..., first_index]
        )
        cross_positive = cross_coefficient >= 0.0

        # L1 = l_j z_i + l_i z_j - l_i l_j
        # L2 = u_j z_i + u_i z_j - u_i u_j
        # U1 = l_j z_i + u_i z_j - u_i l_j
        # U2 = u_j z_i + l_i z_j - l_i u_j
        lower_1_i = cross_coefficient * lower_j
        lower_1_j = cross_coefficient * jnp.where(cross_positive, lower_i, upper_i)
        lower_1_bias = cross_coefficient * jnp.where(
            cross_positive, -lower_i * lower_j, -upper_i * lower_j
        )
        lower_2_i = cross_coefficient * upper_j
        lower_2_j = cross_coefficient * jnp.where(cross_positive, upper_i, lower_i)
        lower_2_bias = cross_coefficient * jnp.where(
            cross_positive, -upper_i * upper_j, -lower_i * upper_j
        )

        upper_1_i = cross_coefficient * lower_j
        upper_1_j = cross_coefficient * jnp.where(cross_positive, upper_i, lower_i)
        upper_1_bias = cross_coefficient * jnp.where(
            cross_positive, -upper_i * lower_j, -lower_i * lower_j
        )
        upper_2_i = cross_coefficient * upper_j
        upper_2_j = cross_coefficient * jnp.where(cross_positive, lower_i, upper_i)
        upper_2_bias = cross_coefficient * jnp.where(
            cross_positive, -lower_i * upper_j, -upper_i * upper_j
        )

        lower_1_at_center = lower_1_i * center_i + lower_1_j * center_j + lower_1_bias
        lower_2_at_center = lower_2_i * center_i + lower_2_j * center_j + lower_2_bias
        choose_lower_1 = lower_1_at_center >= lower_2_at_center
        selected_lower_i = jnp.where(choose_lower_1, lower_1_i, lower_2_i)
        selected_lower_j = jnp.where(choose_lower_1, lower_1_j, lower_2_j)
        selected_lower_bias = jnp.where(choose_lower_1, lower_1_bias, lower_2_bias)

        upper_1_at_center = upper_1_i * center_i + upper_1_j * center_j + upper_1_bias
        upper_2_at_center = upper_2_i * center_i + upper_2_j * center_j + upper_2_bias
        choose_upper_1 = upper_1_at_center <= upper_2_at_center
        selected_upper_i = jnp.where(choose_upper_1, upper_1_i, upper_2_i)
        selected_upper_j = jnp.where(choose_upper_1, upper_1_j, upper_2_j)
        selected_upper_bias = jnp.where(choose_upper_1, upper_1_bias, upper_2_bias)

        lower_coeff = lower_coeff.at[..., first_index].add(selected_lower_i)
        lower_coeff = lower_coeff.at[..., second_index].add(selected_lower_j)
        lower_bias = lower_bias + jnp.sum(selected_lower_bias, axis=-1)
        upper_coeff = upper_coeff.at[..., first_index].add(selected_upper_i)
        upper_coeff = upper_coeff.at[..., second_index].add(selected_upper_j)
        upper_bias = upper_bias + jnp.sum(selected_upper_bias, axis=-1)

        return AffineBound(
            lower_coeff,
            lower_bias,
            upper_coeff,
            upper_bias,
            source_lower,
            source_upper,
        )

    def product_of_source_plane_and_square(
        self,
        first_coeff: ArrayLike,
        first_bias: ArrayLike,
        squared_coeff: ArrayLike,
        squared_bias: ArrayLike,
        *,
        cubic_relaxation: Literal[
            "grouped_interval", "monomial_interval", "zonotope_remainder"
        ] = "monomial_interval",
    ) -> AffineBound:
        """Bound one affine source plane times the square of another.

        For ``f(z) = a^T z + b`` and ``g(z) = c^T z + d``, expand around
        the source-box center ``z_c``.  With ``xi = z - z_c``, write

        ``f = f_c + a^T xi`` and ``g = g_c + c^T xi``.  Then

        ``f g^2 = f_c g_c^2 + (g_c^2 a + 2 f_c g_c c)^T xi + r(xi)``,

        where ``r`` contains every quadratic and cubic term. All modes keep
        the constant and linear terms exact.  ``"grouped_interval"`` applies
        interval arithmetic to the grouped products in ``r``.
        ``"monomial_interval"`` first aggregates the coefficient of each
        canonical quadratic or cubic source monomial and then ranges that
        monomial. ``"zonotope_remainder"`` optimizes the complete nonlinear
        remainder over the two-dimensional zonotope traced by
        ``(a^T xi, c^T xi)``. Centering keeps all relaxed terms proportional
        to the source radii rather than to an arbitrary source origin. The
        monomial families are aggregated with vectorized JAX operations;
        ``"zonotope_remainder"`` additionally sorts the source generators by
        angle and solves a fixed set of edge polynomials.
        """
        if cubic_relaxation not in (
            "grouped_interval",
            "monomial_interval",
            "zonotope_remainder",
        ):
            raise ValueError(
                "cubic_relaxation must be 'grouped_interval', "
                "'monomial_interval', or 'zonotope_remainder'; "
                f"got {cubic_relaxation!r}."
            )

        first_coeff = jnp.asarray(first_coeff)
        first_bias = jnp.asarray(first_bias)
        squared_coeff = jnp.asarray(squared_coeff)
        squared_bias = jnp.asarray(squared_bias)
        if first_coeff.ndim == 0 or first_coeff.shape[-1] != self.input_size:
            raise ValueError(
                "First affine-plane coefficient shape must end with input_size; "
                f"expected {self.input_size}, got {first_coeff.shape}."
            )
        if squared_coeff.ndim == 0 or squared_coeff.shape[-1] != self.input_size:
            raise ValueError(
                "Squared affine-plane coefficient shape must end with input_size; "
                f"expected {self.input_size}, got {squared_coeff.shape}."
            )

        output_shape = jnp.broadcast_shapes(
            first_coeff.shape[:-1],
            first_bias.shape,
            squared_coeff.shape[:-1],
            squared_bias.shape,
        )
        coeff_shape = output_shape + (self.input_size,)
        first_coeff = jnp.broadcast_to(first_coeff, coeff_shape)
        squared_coeff = jnp.broadcast_to(squared_coeff, coeff_shape)
        first_bias = jnp.broadcast_to(first_bias, output_shape)
        squared_bias = jnp.broadcast_to(squared_bias, output_shape)

        source_lower = self.domain_lower
        source_upper = self.domain_upper
        source_center = 0.5 * (source_lower + source_upper)
        centered_lower = source_lower - source_center
        centered_upper = source_upper - source_center

        first_center = (
            jnp.sum(first_coeff * source_center, axis=-1) + first_bias
        )
        squared_center = (
            jnp.sum(squared_coeff * source_center, axis=-1) + squared_bias
        )
        constant = first_center * squared_center**2
        linear_coeff = (
            squared_center[..., None] ** 2 * first_coeff
            + 2.0
            * first_center[..., None]
            * squared_center[..., None]
            * squared_coeff
        )

        if self.input_size == 0:
            return AffineBound(
                linear_coeff,
                constant,
                linear_coeff,
                constant,
                source_lower,
                source_upper,
            )

        zero = jnp.zeros(output_shape, dtype=linear_coeff.dtype)
        first_deviation_lower = self._minimum(
            first_coeff, zero, centered_lower, centered_upper
        )
        first_deviation_upper = self._maximum(
            first_coeff, zero, centered_lower, centered_upper
        )
        squared_deviation_lower = self._minimum(
            squared_coeff, zero, centered_lower, centered_upper
        )
        squared_deviation_upper = self._maximum(
            squared_coeff, zero, centered_lower, centered_upper
        )

        squared_deviation_squared_lower = jnp.where(
            (squared_deviation_lower <= 0.0)
            & (squared_deviation_upper >= 0.0),
            0.0,
            jnp.minimum(
                squared_deviation_lower**2,
                squared_deviation_upper**2,
            ),
        )
        squared_deviation_squared_upper = jnp.maximum(
            squared_deviation_lower**2,
            squared_deviation_upper**2,
        )

        if cubic_relaxation == "zonotope_remainder":
            # The complete nonlinear remainder depends only on
            #
            #   u = first_coeff^T xi,  v = squared_coeff^T xi
            #
            # through
            #
            #   r(u, v) = first_center v^2
            #             + 2 squared_center u v + u v^2.
            #
            # The feasible (u, v) form a centrally symmetric 2D zonotope.
            # Construct its polygonal boundary from angle-sorted generators,
            # optimize the cubic along every edge, and include both possible
            # interior stationary values.
            edge_start, steps, _ = _bilinear_zonotope_edges(
                first_coeff,
                squared_coeff,
                0.5 * (source_upper - source_lower),
            )
            u0 = edge_start[..., 0]
            v0 = edge_start[..., 1]
            du = steps[..., 0]
            dv = steps[..., 1]
            first_center_edge = first_center[..., None]
            squared_center_edge = squared_center[..., None]

            # r(t) = r0 + r1 t + r2 t^2 + r3 t^3 on each edge.
            r0 = (
                first_center_edge * v0**2
                + 2.0 * squared_center_edge * u0 * v0
                + u0 * v0**2
            )
            r1 = (
                2.0 * first_center_edge * v0 * dv
                + 2.0 * squared_center_edge * (u0 * dv + du * v0)
                + 2.0 * u0 * v0 * dv
                + du * v0**2
            )
            r2 = (
                first_center_edge * dv**2
                + 2.0 * squared_center_edge * du * dv
                + u0 * dv**2
                + 2.0 * du * v0 * dv
            )
            r3 = du * dv**2

            derivative_a = 3.0 * r3
            derivative_b = 2.0 * r2
            derivative_c = r1
            is_quadratic = derivative_a != 0.0
            safe_a = jnp.where(is_quadratic, derivative_a, 1.0)
            discriminant = derivative_b**2 - 4.0 * derivative_a * derivative_c
            has_quadratic_roots = is_quadratic & (discriminant >= 0.0)
            root_scale = jnp.sqrt(jnp.maximum(discriminant, 0.0))
            root_minus = (-derivative_b - root_scale) / (2.0 * safe_a)
            root_plus = (-derivative_b + root_scale) / (2.0 * safe_a)

            is_linear = (~is_quadratic) & (derivative_b != 0.0)
            safe_b = jnp.where(is_linear, derivative_b, 1.0)
            root_linear = -derivative_c / safe_b
            roots = jnp.stack((root_minus, root_plus, root_linear), axis=-1)
            root_valid = jnp.stack(
                (
                    has_quadratic_roots,
                    has_quadratic_roots,
                    is_linear,
                ),
                axis=-1,
            ) & (roots >= 0.0) & (roots <= 1.0)
            roots = jnp.clip(roots, 0.0, 1.0)
            root_values = (
                r0[..., None]
                + roots
                * (
                    r1[..., None]
                    + roots
                    * (r2[..., None] + roots * r3[..., None])
                )
            )
            positive_infinity = jnp.asarray(jnp.inf, dtype=linear_coeff.dtype)
            negative_infinity = -positive_infinity
            root_minimum = jnp.min(
                jnp.where(root_valid, root_values, positive_infinity),
                axis=(-2, -1),
            )
            root_maximum = jnp.max(
                jnp.where(root_valid, root_values, negative_infinity),
                axis=(-2, -1),
            )
            endpoint_minimum = jnp.min(
                jnp.minimum(r0, r0 + r1 + r2 + r3), axis=-1
            )
            endpoint_maximum = jnp.max(
                jnp.maximum(r0, r0 + r1 + r2 + r3), axis=-1
            )
            boundary_minimum = jnp.minimum(endpoint_minimum, root_minimum)
            boundary_maximum = jnp.maximum(endpoint_maximum, root_maximum)

            # The origin is always feasible and stationary. The only other
            # isolated stationary point is
            # (u, v) = (-2 first_center, -2 squared_center).
            stationary_u = -2.0 * first_center
            stationary_v = -2.0 * squared_center
            cross = (
                du * (stationary_v[..., None] - v0)
                - dv * (stationary_u[..., None] - u0)
            )
            floating_tolerance = (
                64.0
                * jnp.finfo(linear_coeff.dtype).eps
                * (
                    1.0
                    + jnp.max(jnp.abs(steps), axis=(-2, -1)) ** 2
                    + jnp.abs(stationary_u)
                    + jnp.abs(stationary_v)
                )
            )
            stationary_inside = jnp.all(
                cross >= -floating_tolerance[..., None], axis=-1
            )
            stationary_inside &= (
                stationary_u >= first_deviation_lower - floating_tolerance
            ) & (
                stationary_u <= first_deviation_upper + floating_tolerance
            )
            stationary_inside &= (
                stationary_v >= squared_deviation_lower - floating_tolerance
            ) & (
                stationary_v <= squared_deviation_upper + floating_tolerance
            )
            stationary_value = (
                first_center * stationary_v**2
                + 2.0
                * squared_center
                * stationary_u
                * stationary_v
                + stationary_u * stationary_v**2
            )
            remainder_lower = jnp.minimum(boundary_minimum, 0.0)
            remainder_upper = jnp.maximum(boundary_maximum, 0.0)
            remainder_lower = jnp.minimum(
                remainder_lower,
                jnp.where(
                    stationary_inside, stationary_value, positive_infinity
                ),
            )
            remainder_upper = jnp.maximum(
                remainder_upper,
                jnp.where(
                    stationary_inside, stationary_value, negative_infinity
                ),
            )
        elif cubic_relaxation == "grouped_interval":
            quadratic_square_lower, quadratic_square_upper = (
                self._interval_product(
                    first_center,
                    first_center,
                    squared_deviation_squared_lower,
                    squared_deviation_squared_upper,
                )
            )
            cross_lower, cross_upper = self._interval_product(
                first_deviation_lower,
                first_deviation_upper,
                squared_deviation_lower,
                squared_deviation_upper,
            )
            quadratic_cross_lower, quadratic_cross_upper = (
                self._interval_product(
                    2.0 * squared_center,
                    2.0 * squared_center,
                    cross_lower,
                    cross_upper,
                )
            )
            cubic_lower, cubic_upper = self._interval_product(
                first_deviation_lower,
                first_deviation_upper,
                squared_deviation_squared_lower,
                squared_deviation_squared_upper,
            )
            remainder_lower = (
                quadratic_square_lower + quadratic_cross_lower + cubic_lower
            )
            remainder_upper = (
                quadratic_square_upper + quadratic_cross_upper + cubic_upper
            )
        else:
            remainder_lower = jnp.zeros(output_shape, dtype=linear_coeff.dtype)
            remainder_upper = jnp.zeros(output_shape, dtype=linear_coeff.dtype)

            def add_monomial(coefficient, monomial_lower, monomial_upper):
                term_lower = coefficient * jnp.where(
                    coefficient >= 0.0, monomial_lower, monomial_upper
                )
                term_upper = coefficient * jnp.where(
                    coefficient >= 0.0, monomial_upper, monomial_lower
                )
                return term_lower, term_upper

            # Build static monomial index sets once, then evaluate every term
            # in each family with one vectorized JAX expression. This keeps the
            # compiled graph compact compared with nested Python arithmetic.
            source_square_lower = jnp.where(
                (centered_lower <= 0.0) & (centered_upper >= 0.0),
                0.0,
                jnp.minimum(centered_lower**2, centered_upper**2),
            )
            source_square_upper = jnp.maximum(
                centered_lower**2, centered_upper**2
            )

            # Quadratic monomials xi_i^2.
            coefficient = (
                first_center[..., None] * squared_coeff**2
                + 2.0
                * squared_center[..., None]
                * first_coeff
                * squared_coeff
            )
            term_lower, term_upper = add_monomial(
                coefficient, source_square_lower, source_square_upper
            )
            remainder_lower += jnp.sum(term_lower, axis=-1)
            remainder_upper += jnp.sum(term_upper, axis=-1)

            # Quadratic monomials xi_i xi_j, i < j.
            first_index, second_index = jnp.triu_indices(self.input_size, k=1)
            product_lower, product_upper = self._interval_product(
                centered_lower[first_index],
                centered_upper[first_index],
                centered_lower[second_index],
                centered_upper[second_index],
            )
            coefficient = (
                2.0
                * first_center[..., None]
                * squared_coeff[..., first_index]
                * squared_coeff[..., second_index]
                + 2.0
                * squared_center[..., None]
                * (
                    first_coeff[..., first_index]
                    * squared_coeff[..., second_index]
                    + first_coeff[..., second_index]
                    * squared_coeff[..., first_index]
                )
            )
            term_lower, term_upper = add_monomial(
                coefficient, product_lower, product_upper
            )
            remainder_lower += jnp.sum(term_lower, axis=-1)
            remainder_upper += jnp.sum(term_upper, axis=-1)

            # Cubic monomials xi_i^3.
            coefficient = first_coeff * squared_coeff**2
            term_lower, term_upper = add_monomial(
                coefficient, centered_lower**3, centered_upper**3
            )
            remainder_lower += jnp.sum(term_lower, axis=-1)
            remainder_upper += jnp.sum(term_upper, axis=-1)

            # Cubic monomials xi_i^2 xi_j, i != j.
            square_index = jnp.asarray(
                [
                    i
                    for i in range(self.input_size)
                    for j in range(self.input_size)
                    if i != j
                ],
                dtype=jnp.int32,
            )
            linear_index = jnp.asarray(
                [
                    j
                    for i in range(self.input_size)
                    for j in range(self.input_size)
                    if i != j
                ],
                dtype=jnp.int32,
            )
            monomial_lower, monomial_upper = self._interval_product(
                source_square_lower[square_index],
                source_square_upper[square_index],
                centered_lower[linear_index],
                centered_upper[linear_index],
            )
            coefficient = (
                first_coeff[..., linear_index]
                * squared_coeff[..., square_index] ** 2
                + 2.0
                * first_coeff[..., square_index]
                * squared_coeff[..., square_index]
                * squared_coeff[..., linear_index]
            )
            term_lower, term_upper = add_monomial(
                coefficient, monomial_lower, monomial_upper
            )
            remainder_lower += jnp.sum(term_lower, axis=-1)
            remainder_upper += jnp.sum(term_upper, axis=-1)

            # Cubic monomials xi_i xi_j xi_k, i < j < k.
            triples = [
                (i, j, k)
                for i in range(self.input_size)
                for j in range(i + 1, self.input_size)
                for k in range(j + 1, self.input_size)
            ]
            triple_index = tuple(
                jnp.asarray([item[axis] for item in triples], dtype=jnp.int32)
                for axis in range(3)
            )
            i, j, k = triple_index
            pair_lower, pair_upper = self._interval_product(
                centered_lower[i],
                centered_upper[i],
                centered_lower[j],
                centered_upper[j],
            )
            monomial_lower, monomial_upper = self._interval_product(
                pair_lower,
                pair_upper,
                centered_lower[k],
                centered_upper[k],
            )
            coefficient = 2.0 * (
                first_coeff[..., i]
                * squared_coeff[..., j]
                * squared_coeff[..., k]
                + first_coeff[..., j]
                * squared_coeff[..., i]
                * squared_coeff[..., k]
                + first_coeff[..., k]
                * squared_coeff[..., i]
                * squared_coeff[..., j]
            )
            term_lower, term_upper = add_monomial(
                coefficient, monomial_lower, monomial_upper
            )
            remainder_lower += jnp.sum(term_lower, axis=-1)
            remainder_upper += jnp.sum(term_upper, axis=-1)

        # Convert the exact linear term from xi = z - z_c back to z.
        linear_bias = constant - jnp.sum(
            linear_coeff * source_center, axis=-1
        )
        return AffineBound(
            linear_coeff,
            linear_bias + remainder_lower,
            linear_coeff,
            linear_bias + remainder_upper,
            source_lower,
            source_upper,
        )

    def eliminate_prefix(self, count: int) -> AffineBound:
        """Eliminate leading sources while retaining the affine suffix.

        For a bound on ``q(x, z)``, this minimizes its lower plane and
        maximizes its upper plane over the first ``count`` source coordinates
        ``x``.  The returned bound remains affine in the suffix ``z``.
        """
        if count < 0 or count >= self.input_size:
            raise ValueError(
                "the eliminated affine prefix must leave at least one source; "
                f"got {count} of {self.input_size}"
            )

        eliminated_lower = self.domain_lower[:count]
        eliminated_upper = self.domain_upper[:count]
        lower_prefix = self.lower_coeff[..., :count]
        upper_prefix = self.upper_coeff[..., :count]
        lower_bias = self.lower_bias + self._minimum(
            lower_prefix,
            jnp.zeros_like(self.lower_bias),
            eliminated_lower,
            eliminated_upper,
        )
        upper_bias = self.upper_bias + self._maximum(
            upper_prefix,
            jnp.zeros_like(self.upper_bias),
            eliminated_lower,
            eliminated_upper,
        )
        return AffineBound(
            self.lower_coeff[..., count:],
            lower_bias,
            self.upper_coeff[..., count:],
            upper_bias,
            self.domain_lower[count:],
            self.domain_upper[count:],
        )

    @staticmethod
    def select(
        condition: ArrayLike,
        if_true: AffineBound,
        if_false: AffineBound,
    ) -> AffineBound:
        """Select complete affine planes with a traced condition."""
        if if_true.domain_lower.shape != if_false.domain_lower.shape:
            raise ValueError("AffineBound source-domain shapes must match.")
        condition = jnp.asarray(condition)
        coeff_condition = condition if condition.ndim == 0 else condition[..., None]
        return AffineBound(
            jnp.where(coeff_condition, if_true.lower_coeff, if_false.lower_coeff),
            jnp.where(condition, if_true.lower_bias, if_false.lower_bias),
            jnp.where(coeff_condition, if_true.upper_coeff, if_false.upper_coeff),
            jnp.where(condition, if_true.upper_bias, if_false.upper_bias),
            if_true.domain_lower,
            if_true.domain_upper,
        )

    def _crossing_abs_secant(self, coeff, bias):
        """Return a secant upper bound on an affine plane's absolute value."""
        lower, upper = self.plane_extrema(coeff, bias)
        width = upper - lower
        safe_width = jnp.where(width > 0, width, 1)
        slope = (jnp.abs(upper) - jnp.abs(lower)) / safe_width
        intercept = jnp.abs(lower) - slope * lower
        return slope[..., None] * coeff, slope * bias + intercept, lower, upper

    def _min_lower_planes(self, first_coeff, first_bias, second_coeff, second_bias):
        difference_coeff = first_coeff - second_coeff
        difference_bias = first_bias - second_bias
        abs_coeff, abs_bias, difference_lower, difference_upper = (
            self._crossing_abs_secant(difference_coeff, difference_bias)
        )
        crossing_coeff = 0.5 * (first_coeff + second_coeff - abs_coeff)
        crossing_bias = 0.5 * (first_bias + second_bias - abs_bias)

        first_is_smaller = difference_upper <= 0
        second_is_smaller = difference_lower >= 0
        coeff = jnp.where(
            first_is_smaller[..., None],
            first_coeff,
            jnp.where(second_is_smaller[..., None], second_coeff, crossing_coeff),
        )
        bias = jnp.where(
            first_is_smaller,
            first_bias,
            jnp.where(second_is_smaller, second_bias, crossing_bias),
        )
        return coeff, bias

    def _max_upper_planes(self, first_coeff, first_bias, second_coeff, second_bias):
        difference_coeff = first_coeff - second_coeff
        difference_bias = first_bias - second_bias
        abs_coeff, abs_bias, difference_lower, difference_upper = (
            self._crossing_abs_secant(difference_coeff, difference_bias)
        )
        crossing_coeff = 0.5 * (first_coeff + second_coeff + abs_coeff)
        crossing_bias = 0.5 * (first_bias + second_bias + abs_bias)

        second_is_larger = difference_upper <= 0
        first_is_larger = difference_lower >= 0
        coeff = jnp.where(
            second_is_larger[..., None],
            second_coeff,
            jnp.where(first_is_larger[..., None], first_coeff, crossing_coeff),
        )
        bias = jnp.where(
            second_is_larger,
            second_bias,
            jnp.where(first_is_larger, first_bias, crossing_bias),
        )
        return coeff, bias

    def hull(self, other: AffineBound) -> AffineBound:
        """Return an affine outer hull of two bounds on one source box.

        Dominating planes are retained exactly.  Crossing planes are relaxed
        using a secant upper bound on the absolute value of their difference.
        """
        lower_coeff, lower_bias = self._min_lower_planes(
            self.lower_coeff,
            self.lower_bias,
            other.lower_coeff,
            other.lower_bias,
        )
        upper_coeff, upper_bias = self._max_upper_planes(
            self.upper_coeff,
            self.upper_bias,
            other.upper_coeff,
            other.upper_bias,
        )
        return AffineBound(
            lower_coeff,
            lower_bias,
            upper_coeff,
            upper_bias,
            self.domain_lower,
            self.domain_upper,
        )

    def inflate(self, relative: float = 0.00, absolute: float = 1e-14) -> AffineBound:
        """Inflate the pointwise tube without discarding affine dependence."""
        half_relative = 0.5 * relative
        lower_coeff = (
            1.0 + half_relative
        ) * self.lower_coeff - half_relative * self.upper_coeff
        lower_bias = (
            (1.0 + half_relative) * self.lower_bias
            - half_relative * self.upper_bias
            - absolute
        )
        upper_coeff = (
            -half_relative * self.lower_coeff + (1.0 + half_relative) * self.upper_coeff
        )
        upper_bias = (
            -half_relative * self.lower_bias
            + (1.0 + half_relative) * self.upper_bias
            + absolute
        )
        return AffineBound(
            lower_coeff,
            lower_bias,
            upper_coeff,
            upper_bias,
            self.domain_lower,
            self.domain_upper,
        )

    def inclusion_margin(self, other: AffineBound):
        """Return the worst pointwise margin by which this bound contains another."""
        lower_gap, _ = self.plane_extrema(
            other.lower_coeff - self.lower_coeff,
            other.lower_bias - self.lower_bias,
        )
        upper_gap, _ = self.plane_extrema(
            self.upper_coeff - other.upper_coeff,
            self.upper_bias - other.upper_bias,
        )
        return jnp.minimum(jnp.min(lower_gap), jnp.min(upper_gap))

    def strictly_contains(self, other: AffineBound, margin: float = 0.0):
        """Whether this bound strictly contains ``other`` over the source box."""
        return self.inclusion_margin(other) > margin

    def reshape(self, *shape, **kwargs):
        new_lower_bias = self.lower_bias.reshape(*shape, **kwargs)
        new_shape = new_lower_bias.shape
        return AffineBound(
            self.lower_coeff.reshape(new_shape + (self.input_size,)),
            new_lower_bias,
            self.upper_coeff.reshape(new_shape + (self.input_size,)),
            self.upper_bias.reshape(*shape, **kwargs),
            self.domain_lower,
            self.domain_upper,
        )

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        coeff_axes = tuple(axes) + (self.ndim,)
        return AffineBound(
            self.lower_coeff.transpose(coeff_axes),
            self.lower_bias.transpose(axes),
            self.upper_coeff.transpose(coeff_axes),
            self.upper_bias.transpose(axes),
            self.domain_lower,
            self.domain_upper,
        )

    def __getitem__(self, item):
        coeff_item = item if isinstance(item, tuple) else (item,)
        coeff_item = coeff_item + (slice(None),)
        return AffineBound(
            self.lower_coeff[coeff_item],
            self.lower_bias[item],
            self.upper_coeff[coeff_item],
            self.upper_bias[item],
            self.domain_lower,
            self.domain_upper,
        )

    def __len__(self):
        return len(self.lower_bias)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self):
        return f"AffineBound({self.lower} <= x <= {self.upper})"

    __str__ = __repr__

    # These are installed by aif.py to avoid a circular import.
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    def __pow__(self, exponent): ...
    def __neg__(self): ...
    def __abs__(self): ...


def affine_bound(
    lower_coeff: ArrayLike,
    lower_bias: ArrayLike,
    upper_coeff: ArrayLike,
    upper_bias: ArrayLike,
    domain_lower: ArrayLike,
    domain_upper: ArrayLike,
) -> AffineBound:
    """Construct an :class:`AffineBound` from coefficient and bias arrays."""
    return AffineBound(
        lower_coeff,
        lower_bias,
        upper_coeff,
        upper_bias,
        domain_lower,
        domain_upper,
    )


def constant_affine_bound(
    value: ArrayLike, domain_lower: ArrayLike, domain_upper: ArrayLike
) -> AffineBound:
    """Represent a constant tensor in an existing affine input domain."""
    value = jnp.asarray(value)
    n = jnp.asarray(domain_lower).size
    coeff = jnp.zeros(value.shape + (n,), dtype=jnp.result_type(value, float))
    return AffineBound(coeff, value, coeff, value, domain_lower, domain_upper)


def interval_to_affine_bound(
    value: Interval,
    domain_lower: ArrayLike | None = None,
    domain_upper: ArrayLike | None = None,
    *,
    offset: int = 0,
) -> AffineBound:
    """Lift an interval to exact original-coordinate affine variables.

    When no explicit domain is supplied, the interval itself is flattened and
    used as the complete original input box.
    """
    if domain_lower is None:
        domain_lower = value.lower.reshape(-1)
    if domain_upper is None:
        domain_upper = value.upper.reshape(-1)
    domain_lower = jnp.asarray(domain_lower).reshape(-1)
    domain_upper = jnp.asarray(domain_upper).reshape(-1)
    n = domain_lower.size
    m = value.size
    coeff = jnp.zeros((m, n), dtype=jnp.result_type(value.lower, float))
    coeff = coeff.at[jnp.arange(m), offset + jnp.arange(m)].set(1)
    coeff = coeff.reshape(value.shape + (n,))
    bias = jnp.zeros(value.shape, dtype=jnp.result_type(value.lower, float))
    return AffineBound(coeff, bias, coeff, bias, domain_lower, domain_upper)


def affine_bound_to_interval(value: AffineBound) -> Interval:
    """Concretize an affine bound over its stored original input box."""
    return value.concretize()


__all__ = [
    "AffineBound",
    "affine_bound",
    "affine_bound_to_interval",
    "constant_affine_bound",
    "interval_to_affine_bound",
]
