"""Focused checks for source-optimized affine multiplication."""

import importlib

import jax
import jax.numpy as jnp

import immrax as irx


aif = importlib.import_module("immrax.inclusion.aif")


def _assert_same_bound(first, second):
    assert jnp.allclose(first.lower_coeff, second.lower_coeff)
    assert jnp.allclose(first.lower_bias, second.lower_bias)
    assert jnp.allclose(first.upper_coeff, second.upper_coeff)
    assert jnp.allclose(first.upper_bias, second.upper_bias)


def _nonthin_bound(exact_coeff, exact_bias, lower_shift, upper_shift):
    domain_lower = -jnp.ones(2)
    domain_upper = jnp.ones(2)
    template = irx.constant_affine_bound(0.0, domain_lower, domain_upper)
    lower_coeff = exact_coeff + lower_shift
    upper_coeff = exact_coeff + upper_shift
    _, lower_difference_maximum = template.plane_extrema(
        lower_coeff - exact_coeff, 0.0
    )
    upper_difference_minimum, _ = template.plane_extrema(
        upper_coeff - exact_coeff, 0.0
    )
    return irx.AffineBound(
        lower_coeff,
        exact_bias - lower_difference_maximum - 0.1,
        upper_coeff,
        exact_bias - upper_difference_minimum + 0.1,
        domain_lower,
        domain_upper,
    )


def test_source_optimized_multiplication_is_default():
    assert aif._MUL_RELAXATION == "source_optimized"


def test_multiplication_switch_preserves_baseline_and_selects_experiment(monkeypatch):
    first = _nonthin_bound(
        jnp.array([0.8, -0.4]),
        -0.1,
        jnp.array([0.2, -0.1]),
        jnp.array([-0.15, 0.25]),
    )
    second = _nonthin_bound(
        jnp.array([-0.3, 0.9]),
        0.2,
        jnp.array([-0.1, 0.2]),
        jnp.array([0.3, -0.05]),
    )

    monkeypatch.setattr(aif, "_MUL_RELAXATION", "baseline")
    _assert_same_bound(aif._mul(first, second), aif._mul_mccormick_baseline(
        first, second, first
    ))

    monkeypatch.setattr(aif, "_MUL_RELAXATION", "source_optimized")
    _assert_same_bound(
        aif._mul(first, second),
        aif._mul_mccormick_source_optimized(first, second, first),
    )


def test_source_optimized_multiplication_is_jittable_and_sound_for_mixed_signs():
    first_exact_coeff = jnp.array([0.8, -0.4])
    first_exact_bias = -0.1
    second_exact_coeff = jnp.array([-0.3, 0.9])
    second_exact_bias = 0.2
    first = _nonthin_bound(
        first_exact_coeff,
        first_exact_bias,
        jnp.array([0.2, -0.1]),
        jnp.array([-0.15, 0.25]),
    )
    second = _nonthin_bound(
        second_exact_coeff,
        second_exact_bias,
        jnp.array([-0.1, 0.2]),
        jnp.array([0.3, -0.05]),
    )

    product = jax.jit(aif._mul_mccormick_source_optimized)(
        first, second, first
    )
    axis = jnp.linspace(-1.0, 1.0, 31)
    points = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    exact = (points @ first_exact_coeff + first_exact_bias) * (
        points @ second_exact_coeff + second_exact_bias
    )
    lower = points @ product.lower_coeff + product.lower_bias
    upper = points @ product.upper_coeff + product.upper_bias

    assert jnp.all(lower <= exact + 2e-6)
    assert jnp.all(upper >= exact - 2e-6)


def test_source_optimized_multiplication_dominates_mccormick_vertices():
    first = _nonthin_bound(
        jnp.array([0.8, -0.4]),
        -0.1,
        jnp.array([0.2, -0.1]),
        jnp.array([-0.15, 0.25]),
    )
    second = _nonthin_bound(
        jnp.array([-0.3, 0.9]),
        0.2,
        jnp.array([-0.1, 0.2]),
        jnp.array([0.3, -0.05]),
    )
    product = aif._mul_mccormick_source_optimized(first, second, first)
    pl, pu, ql, qu = jnp.broadcast_arrays(
        first.lower, first.upper, second.lower, second.upper
    )
    lower_vertices = (
        aif._affine_lower_plane(((ql, first), (pl, second)), -pl * ql, first),
        aif._affine_lower_plane(((qu, first), (pu, second)), -pu * qu, first),
    )
    upper_vertices = (
        aif._affine_upper_plane(((ql, first), (pu, second)), -pu * ql, first),
        aif._affine_upper_plane(((qu, first), (pl, second)), -pl * qu, first),
    )
    lower_endpoints = jnp.stack(
        [first.plane_extrema(*plane)[0] for plane in lower_vertices]
    )
    upper_endpoints = jnp.stack(
        [first.plane_extrema(*plane)[1] for plane in upper_vertices]
    )

    assert product.lower >= jnp.max(lower_endpoints) - 2e-6
    assert product.upper <= jnp.min(upper_endpoints) + 2e-6


def test_source_optimized_multiplication_supports_an_empty_source_domain():
    empty = jnp.empty((0,))
    first = irx.AffineBound(empty, 1.0, empty, 2.0, empty, empty)
    second = irx.AffineBound(empty, -3.0, empty, 4.0, empty, empty)

    product = aif._mul_mccormick_source_optimized(first, second, first)

    assert jnp.allclose(product.lower, -6.0)
    assert jnp.allclose(product.upper, 8.0)
