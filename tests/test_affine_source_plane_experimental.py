"""Focused checks for the separate adaptive source-plane experiment."""

import immrax as irx
import jax
import jax.numpy as jnp

from immrax.inclusion.affine_source_plane_experimental import (
    adaptive_zonotope_source_plane_product,
)


def test_adaptive_source_plane_product_is_jittable_and_contains_signed_samples():
    template = irx.constant_affine_bound(
        0.0, jnp.array([-1.0, -2.0]), jnp.array([1.0, 2.0])
    )
    first_coeff = jnp.array([[1.0, 0.4], [-0.7, 0.3]])
    first_bias = jnp.array([0.2, -0.6])
    second_coeff = jnp.array([[-0.5, 0.8], [0.9, -0.2]])
    second_bias = jnp.array([0.4, 0.1])
    bound = jax.jit(adaptive_zonotope_source_plane_product)(
        template, first_coeff, first_bias, second_coeff, second_bias
    )

    axes = (jnp.linspace(-1.0, 1.0, 17), jnp.linspace(-2.0, 2.0, 17))
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    exact = (points @ first_coeff.T + first_bias) * (
        points @ second_coeff.T + second_bias
    )
    lower = points @ bound.lower_coeff.T + bound.lower_bias
    upper = points @ bound.upper_coeff.T + bound.upper_bias
    assert bound.shape == (2,)
    assert jnp.all(lower <= exact + 2e-6)
    assert jnp.all(upper >= exact - 2e-6)


def test_adaptive_source_plane_product_supports_an_empty_source_domain():
    empty = jnp.empty((0,))
    template = irx.constant_affine_bound(0.0, empty, empty)

    bound = adaptive_zonotope_source_plane_product(
        template, empty, 2.0, empty, -3.0
    )

    assert bound.input_size == 0
    assert jnp.allclose(bound.lower, -6.0)
    assert jnp.allclose(bound.upper, -6.0)
