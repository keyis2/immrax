"""Focused sampled checks for the experimental degree-2 Taylor domain."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    TaylorModel,
    evaluate_polynomial,
    normalized_taylor_seed,
    taylor_range,
    taylor_to_affine,
    tmif,
)

jax.config.update("jax_enable_x64", True)


def _sources(count=17):
    axis = np.linspace(-1.0, 1.0, count)
    return np.asarray(list(itertools.product(axis, axis)))


def _assert_contains(bounds, samples, tolerance=1e-12):
    np.testing.assert_array_less(
        np.asarray(bounds.lower) - tolerance,
        np.min(samples, axis=0) + 2 * tolerance,
    )
    np.testing.assert_array_less(
        np.max(samples, axis=0) - 2 * tolerance,
        np.asarray(bounds.upper) + tolerance,
    )


def test_degree_two_product_contains_samples_with_remainders():
    domain_lower, domain_upper = -jnp.ones(2), jnp.ones(2)
    first = TaylorModel(
        0.7,
        jnp.array([0.2, -0.1]),
        jnp.array([[0.3, -0.08], [-0.08, -0.2]]),
        Interval(jnp.array(-0.015), jnp.array(0.025)),
        domain_lower,
        domain_upper,
    )
    second = TaylorModel(
        -0.4,
        jnp.array([0.15, 0.3]),
        jnp.array([[-0.1, 0.12], [0.12, 0.16]]),
        Interval(jnp.array(-0.02), jnp.array(0.01)),
        domain_lower,
        domain_upper,
    )
    product = first * second
    source_samples = _sources()
    remainder_corners = itertools.product(
        [float(first.remainder.lower), float(first.remainder.upper)],
        [float(second.remainder.lower), float(second.remainder.upper)],
    )
    samples = []
    first_poly = jax.vmap(lambda z: evaluate_polynomial(first, z))(source_samples)
    second_poly = jax.vmap(lambda z: evaluate_polynomial(second, z))(source_samples)
    for first_r, second_r in remainder_corners:
        samples.append((first_poly + first_r) * (second_poly + second_r))
    _assert_contains(taylor_range(product), np.concatenate(samples))


def test_sin_and_cos_rules_contain_samples():
    seed = normalized_taylor_seed(jnp.array([-0.35, -0.2]), jnp.array([0.25, 0.3]))
    argument = tmif(lambda z: 0.4 + z[0] - 0.3 * z[1] + z[0] * z[1])(seed)
    sine = tmif(jnp.sin)(argument)
    cosine = tmif(jnp.cos)(argument)
    source_samples = _sources()
    argument_samples = jax.vmap(lambda z: evaluate_polynomial(argument, z))(
        source_samples
    )
    _assert_contains(taylor_range(sine), np.sin(np.asarray(argument_samples)))
    _assert_contains(taylor_range(cosine), np.cos(np.asarray(argument_samples)))


def test_positive_sqrt_and_power_rules_contain_samples():
    seed = normalized_taylor_seed(jnp.array([-0.2, -0.15]), jnp.array([0.2, 0.25]))
    argument = tmif(lambda z: 1.4 + 0.2 * z[0] + z[1] ** 2)(seed)
    square_root = tmif(jnp.sqrt)(argument)
    negative_power = tmif(lambda value: value**-0.5)(argument)
    source_samples = _sources()
    argument_samples = jax.vmap(lambda z: evaluate_polynomial(argument, z))(
        source_samples
    )
    _assert_contains(taylor_range(square_root), np.sqrt(np.asarray(argument_samples)))
    _assert_contains(taylor_range(negative_power), np.asarray(argument_samples) ** -0.5)


def test_vector_range_and_one_time_affine_conversion_contain_polynomial():
    seed = normalized_taylor_seed(jnp.array([-0.2, -0.4]), jnp.array([0.3, 0.1]))
    value = tmif(lambda z: jnp.stack((z[0] * z[1] + z[0], z[0] ** 2 - 0.5 * z[1])))(
        seed
    )
    source_samples = _sources()
    samples = np.asarray(
        jax.vmap(lambda z: evaluate_polynomial(value, z))(source_samples)
    )
    _assert_contains(taylor_range(value), samples)

    affine = taylor_to_affine(value)
    lower_planes = np.einsum(
        "pn,...n->p...", source_samples, np.asarray(affine.lower_coeff)
    ) + np.asarray(affine.lower_bias)
    upper_planes = np.einsum(
        "pn,...n->p...", source_samples, np.asarray(affine.upper_coeff)
    ) + np.asarray(affine.upper_bias)
    assert np.all(lower_planes <= samples + 1e-12)
    assert np.all(samples <= upper_planes + 1e-12)


def test_dot_general_with_two_taylor_operands_contains_samples():
    lower, upper = jnp.array([-0.2, -0.3]), jnp.array([0.25, 0.15])
    seed = normalized_taylor_seed(lower, upper)

    def function(z):
        matrix = jnp.array([[z[0], z[1]], [1.0 + z[1], z[0] * z[1]]])
        vector = jnp.array([z[0], 2.0 - z[1]])
        return matrix @ vector

    result = tmif(function)(seed)
    source_samples = _sources()
    center, radius = (lower + upper) / 2, (upper - lower) / 2
    physical_samples = center + source_samples * radius
    samples = np.asarray(jax.vmap(function)(physical_samples))
    _assert_contains(taylor_range(result), samples)


def test_dead_unsupported_cholesky_does_not_block_live_output():
    seed = normalized_taylor_seed(jnp.array([-0.1, -0.2]), jnp.array([0.2, 0.3]))

    def function(z):
        matrix = jnp.array([[2.0 + z[0] ** 2, 0.1], [0.1, 1.5 + z[1] ** 2]])
        unused_factor = jnp.linalg.cholesky(matrix)
        del unused_factor
        return z[0] * z[1] + z[0]

    result = tmif(function)(seed)
    bounds = taylor_range(result)
    source_samples = _sources()
    center = jnp.array([0.05, 0.05])
    radius = jnp.array([0.15, 0.25])
    samples = np.asarray(jax.vmap(function)(center + source_samples * radius))
    _assert_contains(bounds, samples)


def test_source_dependent_division_contains_samples():
    lower, upper = jnp.array([-0.15, -0.2]), jnp.array([0.2, 0.25])
    seed = normalized_taylor_seed(lower, upper)

    def function(z):
        numerator = 0.7 + z[0] * z[1]
        denominator = 1.4 + 0.2 * z[0] - 0.1 * z[1]
        return numerator / denominator

    result = tmif(function)(seed)
    source_samples = _sources()
    center, radius = (lower + upper) / 2, (upper - lower) / 2
    samples = np.asarray(jax.vmap(function)(center + source_samples * radius))
    _assert_contains(taylor_range(result), samples)


def test_array_literal_subtraction_is_supported():
    lower, upper = jnp.array([-0.1, -0.2]), jnp.array([0.2, 0.3])
    seed = normalized_taylor_seed(lower, upper)
    result = tmif(lambda z: z - jnp.array([1.0, 2.0]))(seed)
    bounds = taylor_range(result)
    np.testing.assert_allclose(bounds.lower, lower - jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(bounds.upper, upper - jnp.array([1.0, 2.0]))


def test_composite_pointwise_errors_stay_in_remainder():
    domain_lower, domain_upper = -jnp.ones(2), jnp.ones(2)
    argument = TaylorModel(
        jnp.array([0.4, -0.25]),
        jnp.array([[0.15, -0.08], [0.06, 0.12]]),
        jnp.array(
            [
                [[0.04, -0.02], [-0.02, 0.03]],
                [[-0.02, 0.01], [0.01, 0.05]],
            ]
        ),
        Interval(jnp.array([-0.015, -0.01]), jnp.array([0.02, 0.018])),
        domain_lower,
        domain_upper,
    )

    def composite(value):
        return jnp.sin((value[0] + 0.3) * (value[1] - 0.2))

    result = tmif(composite)(argument)
    source_samples = _sources(count=13)
    retained_samples = jax.vmap(lambda source: evaluate_polynomial(result, source))(
        source_samples
    )
    input_polynomial_samples = jax.vmap(
        lambda source: evaluate_polynomial(argument, source)
    )(source_samples)

    pointwise_errors = []
    for remainder_value in itertools.product(
        *zip(
            np.asarray(argument.remainder.lower),
            np.asarray(argument.remainder.upper),
        )
    ):
        exact_samples = jax.vmap(composite)(
            input_polynomial_samples + jnp.asarray(remainder_value)
        )
        pointwise_errors.append(exact_samples - retained_samples)
    pointwise_errors = jnp.concatenate(pointwise_errors)

    assert bool(
        jnp.all(pointwise_errors >= result.remainder.lower - 1e-12)
        & jnp.all(pointwise_errors <= result.remainder.upper + 1e-12)
    )
