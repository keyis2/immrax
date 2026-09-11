"""Affine lower/upper bounds expressed in the original interval inputs."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from .interval import Interval


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
        quadratic_relaxation: Literal["affine", "interval", "best"] = "interval",
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
        biases instead.  ``quadratic_relaxation="best"`` constructs both and
        retains the complete lower and upper planes with the tighter
        concretized endpoints.  The mode is a static tracing choice; all
        paths use vectorized JAX operations.
        """
        if quadratic_relaxation == "best":
            affine = self.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="affine",
            )
            interval = self.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="interval",
            )
            use_affine_lower = affine.lower >= interval.lower
            use_affine_upper = affine.upper <= interval.upper
            return AffineBound(
                jnp.where(
                    use_affine_lower[..., None],
                    affine.lower_coeff,
                    interval.lower_coeff,
                ),
                jnp.where(use_affine_lower, affine.lower_bias, interval.lower_bias),
                jnp.where(
                    use_affine_upper[..., None],
                    affine.upper_coeff,
                    interval.upper_coeff,
                ),
                jnp.where(use_affine_upper, affine.upper_bias, interval.upper_bias),
                self.domain_lower,
                self.domain_upper,
            )
        if quadratic_relaxation not in ("affine", "interval"):
            raise ValueError(
                "quadratic_relaxation must be 'affine', 'interval', or 'best'; "
                f"got {quadratic_relaxation!r}."
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
