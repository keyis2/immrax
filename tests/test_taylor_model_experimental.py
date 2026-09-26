"""Focused sampled checks for the experimental degree-2 Taylor domain."""

import itertools
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion.interval import Interval
from immrax.inclusion import taylor as taylor_module
from immrax.inclusion.taylor import (
    TaylorModel,
    _square,
    append_taylor_sources,
    evaluate_polynomial,
    inflate_taylor_remainder,
    interval_taylor_model,
    lift_taylor_remainder_to_sources,
    marginalize_taylor_sources,
    normalized_taylor_seed,
    polynomial_range,
    select_taylor_model,
    taylor_endpoint_models,
    taylor_hull,
    taylor_inclusion_margins,
    taylor_range,
    taylor_relu,
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


def _vector_model():
    return TaylorModel(
        jnp.array([0.5, -0.25]),
        jnp.array([[0.2, -0.1], [0.05, 0.3]]),
        jnp.array(
            [
                [[0.12, -0.04], [0.02, -0.08]],
                [[-0.1, 0.03], [0.07, 0.16]],
            ]
        ),
        Interval(jnp.array([-0.1, -0.2]), jnp.array([0.3, 0.05])),
    )


def _scalar_model(constant, linear, quadratic=0.0, remainder=(0.0, 0.0)):
    return TaylorModel(
        jnp.asarray(constant),
        jnp.asarray([linear]),
        jnp.asarray([[quadratic]]),
        Interval(jnp.asarray(remainder[0]), jnp.asarray(remainder[1])),
    )


def _assert_scaled_model(actual, original, scale):
    scale = np.broadcast_to(np.asarray(scale), original.shape)
    np.testing.assert_allclose(actual.constant, original.constant * scale)
    np.testing.assert_allclose(actual.linear, original.linear * scale[..., None])
    np.testing.assert_allclose(
        actual.quadratic, original.quadratic * scale[..., None, None]
    )
    lower_products = np.asarray(original.remainder.lower) * scale
    upper_products = np.asarray(original.remainder.upper) * scale
    np.testing.assert_allclose(
        actual.remainder.lower, np.minimum(lower_products, upper_products)
    )
    np.testing.assert_allclose(
        actual.remainder.upper, np.maximum(lower_products, upper_products)
    )


def _assert_model_allclose(actual, expected):
    np.testing.assert_allclose(actual.constant, expected.constant)
    np.testing.assert_allclose(actual.linear, expected.linear)
    np.testing.assert_allclose(actual.quadratic, expected.quadratic)
    np.testing.assert_allclose(actual.remainder.lower, expected.remainder.lower)
    np.testing.assert_allclose(actual.remainder.upper, expected.remainder.upper)


def _assert_top(value):
    bounds = taylor_range(value)
    assert bool(jnp.all(jnp.isneginf(bounds.lower)))
    assert bool(jnp.all(jnp.isposinf(bounds.upper)))
    assert bool(jnp.all(value.constant == 0))
    assert bool(jnp.all(value.linear == 0))
    assert bool(jnp.all(value.quadratic == 0))


def _assert_interval_only(value, lower, upper):
    assert bool(jnp.all(value.constant == 0))
    assert bool(jnp.all(value.linear == 0))
    assert bool(jnp.all(value.quadratic == 0))
    np.testing.assert_allclose(value.remainder.lower, lower)
    np.testing.assert_allclose(value.remainder.upper, upper)


def _assert_unary_pointwise_contains(result, argument, function):
    sources = jnp.linspace(-1.0, 1.0, 41).reshape(-1, 1)
    argument_polynomial = jax.vmap(
        lambda source: evaluate_polynomial(argument, source)
    )(sources)
    result_polynomial = jax.vmap(
        lambda source: evaluate_polynomial(result, source)
    )(sources)
    errors = []
    for remainder in np.linspace(
        float(argument.remainder.lower), float(argument.remainder.upper), 11
    ):
        exact = function(argument_polynomial + remainder)
        errors.append(exact - result_polynomial)
    errors = jnp.concatenate(errors)
    assert bool(jnp.all(errors >= result.remainder.lower - 1e-12))
    assert bool(jnp.all(errors <= result.remainder.upper + 1e-12))


def test_normalized_seed_evaluates_to_physical_box_endpoints():
    lower = jnp.array([-0.2, 0.1, -0.4])
    upper = jnp.array([0.3, 0.1, 0.8])
    seed = normalized_taylor_seed(lower, upper)

    assert seed.source_size == 3
    np.testing.assert_allclose(evaluate_polynomial(seed, -jnp.ones(3)), lower)
    np.testing.assert_allclose(evaluate_polynomial(seed, jnp.ones(3)), upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (jnp.array([-0.1, 0.2]), jnp.array([0.3])),
        (jnp.zeros((1, 2)), jnp.ones((1, 2))),
        (jnp.array([]), jnp.array([])),
    ],
)
def test_normalized_seed_rejects_statically_malformed_boxes(lower, upper):
    with pytest.raises(ValueError):
        normalized_taylor_seed(lower, upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (jnp.array([-0.1, jnp.nan]), jnp.array([0.2, 0.3])),
        (jnp.array([-0.1, 0.4]), jnp.array([0.2, 0.3])),
        (jnp.array([-jnp.inf, -0.2]), jnp.array([0.2, 0.3])),
    ],
)
def test_invalid_numerical_seed_is_nan_poisoned_under_jit(lower, upper):
    seed = jax.jit(normalized_taylor_seed)(lower, upper)
    bounds = taylor_range(seed)
    assert bool(jnp.all(jnp.isnan(seed.constant)))
    assert bool(jnp.all(jnp.isnan(seed.linear)))
    assert bool(jnp.all(~jnp.isfinite(bounds.lower)))
    assert bool(jnp.all(~jnp.isfinite(bounds.upper)))


def test_valid_normalized_seed_runs_under_jit():
    lower = jnp.array([-0.2, -0.3])
    upper = jnp.array([0.4, 0.1])
    seed = jax.jit(normalized_taylor_seed)(lower, upper)
    np.testing.assert_allclose(evaluate_polynomial(seed, -jnp.ones(2)), lower)
    np.testing.assert_allclose(evaluate_polynomial(seed, jnp.ones(2)), upper)


def test_unit_box_component_ranges_match_direct_termwise_formula():
    generator = np.random.default_rng(7)
    constant = generator.normal(size=3)
    linear = generator.normal(size=(3, 4))
    quadratic = generator.normal(size=(3, 4, 4))
    zero = jnp.zeros(3)
    value = TaylorModel(
        jnp.asarray(constant),
        jnp.asarray(linear),
        jnp.asarray(quadratic),
        Interval(zero, zero),
    )

    linear_radius = np.sum(np.abs(linear), axis=-1)
    diagonal = 0.5 * np.diagonal(quadratic, axis1=-2, axis2=-1)
    symmetric = 0.5 * (quadratic + np.swapaxes(quadratic, -1, -2))
    off_diagonal_radius = np.sum(
        np.abs(np.triu(symmetric, k=1)), axis=(-2, -1)
    )
    expected_lower = (
        constant
        - linear_radius
        + np.sum(np.minimum(diagonal, 0), axis=-1)
        - off_diagonal_radius
    )
    expected_upper = (
        constant
        + linear_radius
        + np.sum(np.maximum(diagonal, 0), axis=-1)
        + off_diagonal_radius
    )

    actual = polynomial_range(value)
    np.testing.assert_allclose(actual.lower, expected_lower)
    np.testing.assert_allclose(actual.upper, expected_upper)


def test_point_scaling_is_exact_for_scalar_and_broadcast_array():
    value = _vector_model()
    _assert_scaled_model(value * -2.0, value, -2.0)
    _assert_scaled_model(value * jnp.array([2.0, -3.0]), value, [2.0, -3.0])


def test_point_division_reuses_exact_scaling():
    value = _vector_model()
    denominator = jnp.array([2.0, -4.0])
    _assert_scaled_model(value / denominator, value, 1 / denominator)


def test_square_pointwise_remainder_contract_and_strict_tightening():
    value = TaylorModel(
        jnp.array(0.2),
        jnp.array([0.4, -0.3]),
        jnp.array([[0.2, 0.08], [0.08, -0.16]]),
        Interval(jnp.array(-0.25), jnp.array(0.15)),
    )
    square = tmif(lambda operand: operand**2)(value)
    direct_square = _square(value)
    np.testing.assert_allclose(square.constant, direct_square.constant)
    np.testing.assert_allclose(square.linear, direct_square.linear)
    np.testing.assert_allclose(square.quadratic, direct_square.quadratic)
    np.testing.assert_allclose(square.remainder.lower, direct_square.remainder.lower)
    np.testing.assert_allclose(square.remainder.upper, direct_square.remainder.upper)
    independent_product = value * value
    source_samples = _sources(count=15)
    polynomial_samples = jax.vmap(
        lambda source: evaluate_polynomial(value, source)
    )(source_samples)
    retained_square_samples = jax.vmap(
        lambda source: evaluate_polynomial(square, source)
    )(source_samples)
    pointwise_errors = []
    for remainder in np.linspace(-0.25, 0.15, 9):
        pointwise_errors.append(
            (polynomial_samples + remainder) ** 2 - retained_square_samples
        )
    pointwise_errors = jnp.concatenate(pointwise_errors)

    assert bool(
        jnp.all(pointwise_errors >= square.remainder.lower - 1e-12)
        & jnp.all(pointwise_errors <= square.remainder.upper + 1e-12)
    )
    square_width = taylor_range(square).width
    independent_width = taylor_range(independent_product).width
    assert bool(jnp.all(square_width <= independent_width))
    assert bool(jnp.any(square_width < independent_width))


def test_normalized_core_operations_are_jittable():
    value = _vector_model()
    factors = jnp.array([-2.0, 3.0])
    divisors = jnp.array([4.0, -5.0])

    jitted_polynomial = jax.jit(polynomial_range)(value)
    jitted_range = jax.jit(taylor_range)(value)
    jitted_scaled = jax.jit(lambda model, scale: model * scale)(value, factors)
    jitted_divided = jax.jit(lambda model, divisor: model / divisor)(value, divisors)
    jitted_square = jax.jit(_square)(value)

    eager_polynomial = polynomial_range(value)
    eager_range = taylor_range(value)
    np.testing.assert_allclose(jitted_polynomial.lower, eager_polynomial.lower)
    np.testing.assert_allclose(jitted_polynomial.upper, eager_polynomial.upper)
    np.testing.assert_allclose(jitted_range.lower, eager_range.lower)
    np.testing.assert_allclose(jitted_range.upper, eager_range.upper)
    _assert_scaled_model(jitted_scaled, value, factors)
    _assert_scaled_model(jitted_divided, value, 1 / divisors)
    np.testing.assert_allclose(jitted_square.constant, _square(value).constant)
    np.testing.assert_allclose(jitted_square.linear, _square(value).linear)
    np.testing.assert_allclose(jitted_square.quadratic, _square(value).quadratic)
    np.testing.assert_allclose(
        jitted_square.remainder.lower, _square(value).remainder.lower
    )
    np.testing.assert_allclose(
        jitted_square.remainder.upper, _square(value).remainder.upper
    )


def test_relu_active_inactive_crossing_and_mixed_components():
    active = _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08))
    inactive = _scalar_model(-1.5, 0.2, 0.1, (-0.05, 0.08))
    crossing = _scalar_model(0.1, 0.7, 0.2, (-0.15, 0.1))

    _assert_model_allclose(taylor_relu(active), active)
    inactive_relu = taylor_relu(inactive)
    np.testing.assert_allclose(taylor_range(inactive_relu).lower, 0)
    np.testing.assert_allclose(taylor_range(inactive_relu).upper, 0)
    crossing_relu = taylor_relu(crossing)
    crossing_bounds = taylor_range(crossing)
    slope = crossing_bounds.upper / (crossing_bounds.upper - crossing_bounds.lower)
    error_upper = (
        -crossing_bounds.lower
        * crossing_bounds.upper
        / (crossing_bounds.upper - crossing_bounds.lower)
    )
    np.testing.assert_allclose(crossing_relu.constant, slope * crossing.constant)
    np.testing.assert_allclose(crossing_relu.linear, slope * crossing.linear)
    np.testing.assert_allclose(crossing_relu.quadratic, slope * crossing.quadratic)
    np.testing.assert_allclose(
        crossing_relu.remainder.lower, slope * crossing.remainder.lower
    )
    np.testing.assert_allclose(
        crossing_relu.remainder.upper,
        slope * crossing.remainder.upper + error_upper,
    )
    _assert_unary_pointwise_contains(crossing_relu, crossing, jax.nn.relu)

    mixed = TaylorModel(
        jnp.stack((active.constant, inactive.constant, crossing.constant)),
        jnp.stack((active.linear, inactive.linear, crossing.linear)),
        jnp.stack((active.quadratic, inactive.quadratic, crossing.quadratic)),
        Interval(
            jnp.stack(
                (
                    active.remainder.lower,
                    inactive.remainder.lower,
                    crossing.remainder.lower,
                )
            ),
            jnp.stack(
                (
                    active.remainder.upper,
                    inactive.remainder.upper,
                    crossing.remainder.upper,
                )
            ),
        ),
    )
    mixed_relu = taylor_relu(mixed)
    _assert_model_allclose(mixed_relu[0], active)
    np.testing.assert_allclose(taylor_range(mixed_relu[1]).lower, 0)
    np.testing.assert_allclose(taylor_range(mixed_relu[1]).upper, 0)
    _assert_model_allclose(mixed_relu[2], crossing_relu)


def test_jitted_tmif_maximum_handles_all_relu_branches():
    transform = jax.jit(tmif(lambda value: jnp.maximum(value, 0)))
    models = (
        _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08)),
        _scalar_model(-1.5, 0.2, 0.1, (-0.05, 0.08)),
        _scalar_model(0.1, 0.7, 0.2, (-0.15, 0.1)),
    )
    for model in models:
        _assert_model_allclose(transform(model), taylor_relu(model))


def test_maximum_minimum_and_abs_preserve_fixed_branches_and_crossings():
    positive = _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08))
    negative = _scalar_model(-1.5, 0.2, 0.1, (-0.05, 0.08))
    crossing = _scalar_model(0.1, 0.7, 0.2, (-0.15, 0.1))

    maximum = tmif(lambda value: jnp.maximum(value, 0))
    minimum = tmif(lambda value: jnp.minimum(value, 0))
    absolute = tmif(jnp.abs)
    _assert_model_allclose(maximum(positive), positive)
    _assert_model_allclose(minimum(negative), negative)
    _assert_model_allclose(absolute(positive), positive)
    _assert_model_allclose(absolute(negative), -negative)
    _assert_unary_pointwise_contains(maximum(crossing), crossing, jax.nn.relu)
    _assert_unary_pointwise_contains(
        minimum(crossing), crossing, lambda value: jnp.minimum(value, 0)
    )
    _assert_unary_pointwise_contains(absolute(crossing), crossing, jnp.abs)


def test_reciprocal_regular_fallback_and_top_are_eager_and_jittable():
    transform = tmif(lambda value: 1.0 / value)
    jitted = jax.jit(transform)
    regular = _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08))
    fallback_argument = _scalar_model(0.0, 0.0, 0.0, (1.0, 2.0))
    singular = _scalar_model(0.0, 1.0)

    eager_regular = transform(regular)
    _assert_unary_pointwise_contains(eager_regular, regular, jnp.reciprocal)
    _assert_model_allclose(jitted(regular), eager_regular)

    fallback = transform(fallback_argument)
    _assert_interval_only(fallback, 0.5, 1.0)
    _assert_model_allclose(jitted(fallback_argument), fallback)

    _assert_top(transform(singular))
    _assert_top(jitted(singular))


def test_sqrt_regular_boundary_fallback_and_top_are_eager_and_jittable():
    transform = tmif(jnp.sqrt)
    jitted = jax.jit(transform)
    regular = _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08))
    boundary = _scalar_model(0.5, 0.5)
    undefined = _scalar_model(0.0, 1.0)

    eager_regular = transform(regular)
    _assert_unary_pointwise_contains(eager_regular, regular, jnp.sqrt)
    _assert_model_allclose(jitted(regular), eager_regular)

    fallback = transform(boundary)
    _assert_interval_only(fallback, 0.0, 1.0)
    _assert_model_allclose(jitted(boundary), fallback)

    _assert_top(transform(undefined))
    _assert_top(jitted(undefined))


def test_general_power_regular_fallback_and_top_are_jittable():
    transform = tmif(lambda value, exponent: jnp.power(value, exponent))
    jitted = jax.jit(transform)
    regular = _scalar_model(1.5, 0.2, 0.1, (-0.05, 0.08))
    boundary = _scalar_model(0.5, 0.5)
    negative_crossing = _scalar_model(0.0, 1.0)

    exponent = jnp.asarray(0.5)
    eager_regular = transform(regular, exponent)
    _assert_unary_pointwise_contains(
        eager_regular, regular, lambda value: value**exponent
    )
    _assert_model_allclose(jitted(regular, exponent), eager_regular)

    fallback = transform(boundary, exponent)
    _assert_interval_only(fallback, 0.0, 1.0)
    _assert_model_allclose(jitted(boundary, exponent), fallback)

    _assert_top(transform(negative_crossing, exponent))
    _assert_top(jitted(negative_crossing, exponent))
    _assert_top(transform(boundary, jnp.asarray(-0.5)))
    _assert_top(jitted(boundary, jnp.asarray(jnp.nan)))


def test_uncertain_selection_uses_interval_hull_and_is_jittable():
    argument = _scalar_model(0.0, 1.0, 0.0, (-0.1, 0.1))
    point_function = lambda value: jnp.where(value > 0, value + 2, value - 3)
    transform = tmif(point_function)
    result = transform(argument)
    _assert_unary_pointwise_contains(result, argument, point_function)
    _assert_model_allclose(jax.jit(transform)(argument), result)


def test_taylor_core_has_no_host_numpy_numerical_decisions():
    source = Path(taylor_module.__file__).read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert re.search(r"(?<!j)\bnp\.", source) is None


def test_degree_two_product_contains_samples_with_remainders():
    first = TaylorModel(
        0.7,
        jnp.array([0.2, -0.1]),
        jnp.array([[0.3, -0.08], [-0.08, -0.2]]),
        Interval(jnp.array(-0.015), jnp.array(0.025)),
    )
    second = TaylorModel(
        -0.4,
        jnp.array([0.15, 0.3]),
        jnp.array([[-0.1, 0.12], [0.12, 0.16]]),
        Interval(jnp.array(-0.02), jnp.array(0.01)),
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
    np.testing.assert_array_equal(affine.domain_lower, -jnp.ones(value.source_size))
    np.testing.assert_array_equal(affine.domain_upper, jnp.ones(value.source_size))
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


def test_remainder_lift_is_fiber_exact_for_vector_model():
    value = _vector_model()
    lifted = lift_taylor_remainder_to_sources(value)

    assert lifted.source_size == value.source_size + value.size
    np.testing.assert_allclose(lifted.remainder.lower, 0.0)
    np.testing.assert_allclose(lifted.remainder.upper, 0.0)
    source = jnp.array([0.3, -0.7])
    for remainder_source in itertools.product((-1.0, 0.25, 1.0), repeat=2):
        remainder_source = jnp.asarray(remainder_source)
        original_polynomial = evaluate_polynomial(value, source)
        midpoint = 0.5 * (value.remainder.lower + value.remainder.upper)
        radius = 0.5 * (value.remainder.upper - value.remainder.lower)
        expected = original_polynomial + midpoint + radius * remainder_source
        actual = evaluate_polynomial(
            lifted, jnp.concatenate((source, remainder_source))
        )
        np.testing.assert_allclose(actual, expected)


def test_remainder_lift_is_exact_for_scalar_model():
    value = _scalar_model(0.2, 0.4, quadratic=-0.1, remainder=(-0.3, 0.5))
    lifted = lift_taylor_remainder_to_sources(value)
    assert lifted.source_size == 2
    for source, remainder_source in itertools.product(
        np.linspace(-1.0, 1.0, 7), np.linspace(-1.0, 1.0, 7)
    ):
        expected = (
            evaluate_polynomial(value, jnp.asarray([source]))
            + 0.1
            + 0.4 * remainder_source
        )
        actual = evaluate_polynomial(
            lifted, jnp.asarray([source, remainder_source])
        )
        np.testing.assert_allclose(actual, expected)


def test_marginalization_contains_all_eliminated_source_samples():
    value = TaylorModel(
        jnp.array([0.2, -0.4]),
        jnp.array([[0.3, -0.2, 0.4], [-0.1, 0.25, -0.35]]),
        jnp.array(
            [
                [[0.1, 0.12, -0.08], [0.04, -0.2, 0.16], [0.02, -0.06, 0.3]],
                [[-0.2, 0.06, 0.1], [-0.02, 0.14, -0.18], [0.08, 0.04, -0.1]],
            ]
        ),
        Interval(jnp.array([-0.03, -0.04]), jnp.array([0.05, 0.02])),
    )
    marginalized = marginalize_taylor_sources(value, 1)
    kept_sources = jnp.linspace(-1.0, 1.0, 7)
    eliminated_sources = np.asarray(
        list(itertools.product(np.linspace(-1.0, 1.0, 9), repeat=2))
    )
    for kept_source in kept_sources:
        retained = evaluate_polynomial(marginalized, jnp.asarray([kept_source]))
        exact_polynomials = jax.vmap(
            lambda eliminated: evaluate_polynomial(
                value, jnp.concatenate((jnp.asarray([kept_source]), eliminated))
            )
        )(eliminated_sources)
        errors = exact_polynomials - retained
        assert bool(jnp.all(errors >= marginalized.remainder.lower - 1e-12))
        assert bool(jnp.all(errors <= marginalized.remainder.upper + 1e-12))


def test_taylor_hull_contains_both_source_fibers():
    first = _vector_model()
    second = TaylorModel(
        jnp.array([-0.1, 0.4]),
        jnp.array([[0.1, 0.05], [-0.2, 0.15]]),
        jnp.array(
            [[[0.02, 0.06], [-0.01, 0.1]], [[-0.08, 0.02], [0.04, 0.03]]]
        ),
        Interval(jnp.array([-0.08, -0.03]), jnp.array([0.12, 0.09])),
    )
    hull = taylor_hull(first, second)
    for source in _sources(count=9):
        retained = evaluate_polynomial(hull, source)
        for value in (first, second):
            polynomial = evaluate_polynomial(value, source)
            lower_error = polynomial + value.remainder.lower - retained
            upper_error = polynomial + value.remainder.upper - retained
            assert bool(jnp.all(lower_error >= hull.remainder.lower - 1e-12))
            assert bool(jnp.all(upper_error <= hull.remainder.upper + 1e-12))


def test_inflate_select_interval_and_endpoint_helpers_are_jittable():
    seed = normalized_taylor_seed(jnp.array([-0.2]), jnp.array([0.4]))
    interval_only = interval_taylor_model(-0.3, 0.5, seed)
    inflated = jax.jit(
        lambda value: inflate_taylor_remainder(value, relative=0.5, absolute=0.1)
    )(interval_only)
    np.testing.assert_allclose(inflated.remainder.lower, -0.6)
    np.testing.assert_allclose(inflated.remainder.upper, 0.8)

    selected = jax.jit(select_taylor_model)(
        jnp.array(True), inflated, interval_only
    )
    _assert_model_allclose(selected, inflated)
    lower, upper = taylor_endpoint_models(inflated)
    np.testing.assert_allclose(taylor_range(lower).lower, -0.6)
    np.testing.assert_allclose(taylor_range(lower).upper, -0.6)
    np.testing.assert_allclose(taylor_range(upper).lower, 0.8)
    np.testing.assert_allclose(taylor_range(upper).upper, 0.8)


def test_inclusion_margins_match_analytic_affine_case_under_jit():
    outer = TaylorModel(
        jnp.array([0.0]),
        jnp.array([[0.5]]),
        jnp.zeros((1, 1, 1)),
        Interval(jnp.array([-0.4]), jnp.array([0.6])),
    )
    inner = TaylorModel(
        jnp.array([0.1]),
        jnp.array([[0.2]]),
        jnp.zeros((1, 1, 1)),
        Interval(jnp.array([-0.1]), jnp.array([0.2])),
    )
    lower_margin, upper_margin = jax.jit(taylor_inclusion_margins)(outer, inner)
    np.testing.assert_allclose(lower_margin, 0.1)
    np.testing.assert_allclose(upper_margin, 0.0, atol=1e-15)


def test_source_operations_reject_invalid_static_counts_and_shapes():
    value = _vector_model()
    with pytest.raises(ValueError):
        append_taylor_sources(value, -1)
    with pytest.raises(ValueError):
        marginalize_taylor_sources(value, -1)
    with pytest.raises(ValueError):
        marginalize_taylor_sources(value, value.source_size + 1)
    with pytest.raises(ValueError):
        taylor_hull(value, append_taylor_sources(value, 1))
