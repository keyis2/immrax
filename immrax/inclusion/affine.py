"""Affine lower/upper bounds expressed in the original interval inputs."""

from __future__ import annotations

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
