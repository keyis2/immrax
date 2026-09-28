"""Focused contracts for scalar source-dependent ReLU endpoint products."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    _RELU_BRANCH_ACTIVE as RELU_BRANCH_ACTIVE,
    _RELU_BRANCH_CROSSING as RELU_BRANCH_CROSSING,
    _RELU_BRANCH_INACTIVE as RELU_BRANCH_INACTIVE,
    _joint_endpoint_product as joint_endpoint_product,
    _quadratic_product_envelopes as quadratic_product_envelopes,
    QuadraticEndpointPair,
    TaylorModel,
    endpoint_pair_to_taylor,
    evaluate_polynomial,
    taylor_range,
    taylor_relu_product,
)

jax.config.update("jax_enable_x64", True)

_TOLERANCE = 3e-12


def _model(constant, linear, quadratic, remainder):
    return TaylorModel(
        jnp.asarray(constant, dtype=float),
        jnp.asarray(linear, dtype=float),
        jnp.asarray(quadratic, dtype=float),
        Interval(
            jnp.asarray(remainder[0], dtype=float),
            jnp.asarray(remainder[1], dtype=float),
        ),
    )


def _models():
    quadratic = [[0.08, -0.025], [-0.025, -0.05]]
    return {
        "inactive": _model(-1.4, [0.18, -0.11], quadratic, (-0.07, 0.04)),
        "active": _model(1.4, [-0.14, 0.19], quadratic, (-0.05, 0.08)),
        "crossing": _model(
            0.04,
            [0.36, -0.17],
            [[0.10, 0.04], [0.04, -0.06]],
            (-0.08, 0.055),
        ),
    }


def _sources(count=13):
    axis = np.linspace(-1.0, 1.0, count)
    return jnp.asarray(list(itertools.product(axis, repeat=2)))


def _random_sources(rng, source_size, count=512):
    return jnp.asarray(rng.uniform(-1.0, 1.0, size=(count, source_size)))


def _reference_polynomial_coefficients(model):
    source_size = model.source_size
    symmetric = 0.5 * (model.quadratic + model.quadratic.T)
    zero = (0,) * source_size
    coefficients = {zero: model.constant}
    for i in range(source_size):
        alpha = tuple(int(axis == i) for axis in range(source_size))
        coefficients[alpha] = model.linear[i]
    for i in range(source_size):
        for j in range(i, source_size):
            alpha = tuple(
                int(axis == i) + int(axis == j)
                for axis in range(source_size)
            )
            coefficients[alpha] = (
                0.5 * symmetric[i, i] if i == j else symmetric[i, j]
            )
    return coefficients


def _reference_product_coefficients(first, second):
    source_size = first.source_size
    product = {}
    zero = jnp.zeros_like(first.constant)
    for first_alpha, first_coefficient in _reference_polynomial_coefficients(
        first
    ).items():
        for second_alpha, second_coefficient in _reference_polynomial_coefficients(
            second
        ).items():
            alpha = tuple(
                first_alpha[index] + second_alpha[index]
                for index in range(source_size)
            )
            product[alpha] = product.get(alpha, zero) + (
                first_coefficient * second_coefficient
            )
    return product


def _reference_correction(alpha, coefficient):
    degree = sum(alpha)
    weights = jnp.asarray(alpha, dtype=coefficient.dtype) / degree
    if all(exponent % 2 == 0 for exponent in alpha):
        return (
            jnp.minimum(coefficient, 0.0) * weights,
            jnp.maximum(coefficient, 0.0) * weights,
        )
    magnitude = jnp.abs(coefficient) * weights
    return -magnitude, magnitude


def _reference_product_envelopes(first, second):
    source_size = first.source_size
    coefficients = _reference_product_coefficients(first, second)
    zero_alpha = (0,) * source_size
    constant = coefficients[zero_alpha]
    linear = jnp.zeros_like(first.linear)
    lower_quadratic = jnp.zeros_like(first.quadratic)
    upper_quadratic = jnp.zeros_like(first.quadratic)
    for alpha, coefficient in coefficients.items():
        degree = sum(alpha)
        if degree == 1:
            linear = linear.at[alpha.index(1)].add(coefficient)
        elif degree == 2:
            indices = [
                index
                for index, exponent in enumerate(alpha)
                for _ in range(exponent)
            ]
            first_index, second_index = indices
            factor = 2.0 if first_index == second_index else 1.0
            lower_quadratic = lower_quadratic.at[
                first_index, second_index
            ].add(factor * coefficient)
            upper_quadratic = upper_quadratic.at[
                first_index, second_index
            ].add(factor * coefficient)
            if first_index != second_index:
                lower_quadratic = lower_quadratic.at[
                    second_index, first_index
                ].add(coefficient)
                upper_quadratic = upper_quadratic.at[
                    second_index, first_index
                ].add(coefficient)
        elif degree in (3, 4):
            lower, upper = _reference_correction(alpha, coefficient)
            lower_quadratic += jnp.diag(2.0 * lower)
            upper_quadratic += jnp.diag(2.0 * upper)
    zero_remainder = Interval(jnp.zeros_like(constant), jnp.zeros_like(constant))
    return (
        TaylorModel(constant, linear, lower_quadratic, zero_remainder),
        TaylorModel(constant, linear, upper_quadratic, zero_remainder),
    )


def _random_zero_remainder_model(rng, source_size):
    linear = rng.normal(scale=0.25, size=source_size)
    raw_quadratic = rng.normal(scale=0.12, size=(source_size, source_size))
    quadratic = 0.5 * (raw_quadratic + raw_quadratic.T)
    return _model(
        rng.normal(scale=0.3),
        linear,
        quadratic,
        (0.0, 0.0),
    )


def _corner_products(first, second, sources):
    first_polynomial = jax.vmap(
        lambda source: evaluate_polynomial(first, source)
    )(sources)
    second_polynomial = jax.vmap(
        lambda source: evaluate_polynomial(second, source)
    )(sources)
    return jnp.stack(
        [
            jax.nn.relu(first_polynomial + first_remainder)
            * jax.nn.relu(second_polynomial + second_remainder)
            for first_remainder, second_remainder in itertools.product(
                (first.remainder.lower, first.remainder.upper),
                (second.remainder.lower, second.remainder.upper),
            )
        ]
    )


def _assert_contains(first, second):
    pair = joint_endpoint_product(first, second)
    assert bool(pair.preconditions_hold)
    converted = endpoint_pair_to_taylor(pair)
    direct = taylor_relu_product(first, second)
    sources = _sources()
    exact = _corner_products(first, second, sources)
    lower = jax.vmap(lambda source: evaluate_polynomial(pair.lower, source))(
        sources
    )
    upper = jax.vmap(lambda source: evaluate_polynomial(pair.upper, source))(
        sources
    )
    assert bool(jnp.all(exact >= lower[None, :] - _TOLERANCE))
    assert bool(jnp.all(exact <= upper[None, :] + _TOLERANCE))
    for model in (converted, direct):
        polynomial = jax.vmap(
            lambda source: evaluate_polynomial(model, source)
        )(sources)
        assert bool(
            jnp.all(
                exact
                >= polynomial[None, :] + model.remainder.lower - _TOLERANCE
            )
        )
        assert bool(
            jnp.all(
                exact
                <= polynomial[None, :] + model.remainder.upper + _TOLERANCE
            )
        )
    return pair, converted


def test_all_ordered_relu_branch_pairs_and_exact_inactive_zero():
    models = _models()
    codes = {
        "inactive": RELU_BRANCH_INACTIVE,
        "active": RELU_BRANCH_ACTIVE,
        "crossing": RELU_BRANCH_CROSSING,
    }
    for first_name, second_name in itertools.product(models, repeat=2):
        pair, _converted = _assert_contains(
            models[first_name], models[second_name]
        )
        assert int(pair.first_endpoints.branch_code) == codes[first_name]
        assert int(pair.second_endpoints.branch_code) == codes[second_name]
        if "inactive" in (first_name, second_name):
            for endpoint in (pair.lower, pair.upper):
                assert all(
                    bool(jnp.all(leaf == 0.0))
                    for leaf in jax.tree_util.tree_leaves(endpoint)
                )


def test_active_affine_zero_remainder_product_is_exact():
    zeros = np.zeros((2, 2))
    first = _model(1.3, [0.11, -0.08], zeros, (0.0, 0.0))
    second = _model(0.9, [-0.07, 0.13], zeros, (0.0, 0.0))
    pair, converted = _assert_contains(first, second)
    sources = _sources()
    exact = jax.vmap(
        lambda source: evaluate_polynomial(first, source)
        * evaluate_polynomial(second, source)
    )(sources)
    for model in (pair.lower, pair.upper, converted):
        values = jax.vmap(
            lambda source: evaluate_polynomial(model, source)
        )(sources)
        np.testing.assert_allclose(values, exact, rtol=0.0, atol=_TOLERANCE)
    np.testing.assert_allclose(converted.remainder.width, 0.0)


def test_endpoint_construction_is_directly_jittable_and_vmappable():
    first = _models()
    second = {
        "inactive": first["active"],
        "active": first["crossing"],
        "crossing": first["inactive"],
    }
    first_batch = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *first.values()
    )
    second_batch = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *second.values()
    )
    eager = jax.vmap(joint_endpoint_product)(first_batch, second_batch)
    compiled = jax.jit(jax.vmap(joint_endpoint_product))(
        first_batch, second_batch
    )
    assert eager.lower.shape == (3,)
    assert bool(jnp.all(eager.preconditions_hold))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compiled),
        jax.tree_util.tree_leaves(eager),
    ):
        np.testing.assert_allclose(actual, expected)


def test_invalid_nonfinite_input_returns_status_and_nan_without_fallback():
    invalid = _model(
        jnp.nan,
        [0.1, -0.2],
        [[0.03, 0.0], [0.0, -0.04]],
        (-0.05, 0.07),
    )
    pair = joint_endpoint_product(invalid, _models()["active"])
    assert not bool(pair.preconditions_hold)
    converted = endpoint_pair_to_taylor(pair)
    assert all(
        bool(jnp.all(jnp.isnan(leaf)))
        for leaf in jax.tree_util.tree_leaves(converted)
    )


def _dimension_model(source_size, offset):
    axis = np.linspace(-1.0, 1.0, source_size)
    quadratic = 0.035 * np.outer(axis, axis)
    quadratic += np.diag(np.linspace(-0.025, 0.04, source_size))
    return _model(
        0.05 + offset,
        0.16 * axis[::-1] + 0.01 * offset,
        quadratic,
        (-0.055, 0.045),
    )


def test_direct_construction_has_dimension_stable_jaxpr_without_scatter():
    equation_counts = []
    for source_size in (2, 8):
        first = _dimension_model(source_size, 0.0)
        second = _dimension_model(source_size, 0.07)
        eager = joint_endpoint_product(first, second)
        compiled = jax.jit(joint_endpoint_product)(first, second)
        batch_first = jax.tree_util.tree_map(
            lambda leaf: jnp.stack((leaf, leaf)), first
        )
        batch_second = jax.tree_util.tree_map(
            lambda leaf: jnp.stack((leaf, leaf)), second
        )
        mapped = jax.vmap(joint_endpoint_product)(batch_first, batch_second)
        assert eager.lower.source_size == source_size
        assert mapped.lower.shape == (2,)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(compiled),
            jax.tree_util.tree_leaves(eager),
        ):
            np.testing.assert_allclose(actual, expected)

        jaxpr = jax.make_jaxpr(joint_endpoint_product)(first, second).jaxpr
        equation_counts.append(len(jaxpr.eqns))
        assert "scatter" not in str(jaxpr)

    assert equation_counts[0] == equation_counts[1]


def test_vector_kernel_matches_reference_containment_and_reports_widths():
    rng = np.random.default_rng(20260927)
    deterministic = _models()
    cases = [
        (
            TaylorModel(
                deterministic["active"].constant,
                deterministic["active"].linear,
                deterministic["active"].quadratic,
                Interval(
                    jnp.zeros_like(deterministic["active"].constant),
                    jnp.zeros_like(deterministic["active"].constant),
                ),
            ),
            TaylorModel(
                deterministic["crossing"].constant,
                deterministic["crossing"].linear,
                deterministic["crossing"].quadratic,
                Interval(
                    jnp.zeros_like(deterministic["crossing"].constant),
                    jnp.zeros_like(deterministic["crossing"].constant),
                ),
            ),
        )
    ]
    cases.extend(
        (
            _random_zero_remainder_model(rng, 3),
            _random_zero_remainder_model(rng, 3),
        )
        for _ in range(8)
    )
    raw_global_ratios = []
    raw_mean_ratios = []
    converted_global_ratios = []
    converted_mean_ratios = []
    for first, second in cases:
        sources = _random_sources(rng, first.source_size)
        exact = jax.vmap(
            lambda source: evaluate_polynomial(first, source)
            * evaluate_polynomial(second, source)
        )(sources)
        vector_lower, vector_upper = quadratic_product_envelopes(first, second)
        reference_lower, reference_upper = _reference_product_envelopes(
            first, second
        )
        for lower, upper in (
            (vector_lower, vector_upper),
            (reference_lower, reference_upper),
        ):
            lower_values = jax.vmap(
                lambda source: evaluate_polynomial(lower, source)
            )(sources)
            upper_values = jax.vmap(
                lambda source: evaluate_polynomial(upper, source)
            )(sources)
            assert bool(jnp.all(exact >= lower_values - _TOLERANCE))
            assert bool(jnp.all(exact <= upper_values + _TOLERANCE))

        vector_width = jax.vmap(
            lambda source: evaluate_polynomial(vector_upper, source)
            - evaluate_polynomial(vector_lower, source)
        )(sources)
        reference_width = jax.vmap(
            lambda source: evaluate_polynomial(reference_upper, source)
            - evaluate_polynomial(reference_lower, source)
        )(sources)
        raw_global_ratios.append(
            float(jnp.max(vector_width) / jnp.maximum(jnp.max(reference_width), 1e-15))
        )
        raw_mean_ratios.append(
            float(jnp.mean(vector_width) / jnp.maximum(jnp.mean(reference_width), 1e-15))
        )

        vector_pair = QuadraticEndpointPair(vector_lower, vector_upper, True)
        reference_pair = QuadraticEndpointPair(
            reference_lower, reference_upper, True
        )
        vector_converted = endpoint_pair_to_taylor(vector_pair)
        reference_converted = endpoint_pair_to_taylor(reference_pair)
        for converted in (vector_converted, reference_converted):
            polynomial = jax.vmap(
                lambda source: evaluate_polynomial(converted, source)
            )(sources)
            assert bool(
                jnp.all(
                    exact
                    >= polynomial + converted.remainder.lower - _TOLERANCE
                )
            )
            assert bool(
                jnp.all(
                    exact
                    <= polynomial + converted.remainder.upper + _TOLERANCE
                )
            )
        converted_global_ratios.append(
            float(
                taylor_range(vector_converted).width
                / jnp.maximum(taylor_range(reference_converted).width, 1e-15)
            )
        )
        converted_mean_ratios.append(
            float(
                vector_converted.remainder.width
                / jnp.maximum(reference_converted.remainder.width, 1e-15)
            )
        )

    print(
        "vector/reference max ratios: "
        f"raw_global={max(raw_global_ratios):.6f}, "
        f"raw_mean={max(raw_mean_ratios):.6f}, "
        f"converted_global={max(converted_global_ratios):.6f}, "
        f"converted_mean={max(converted_mean_ratios):.6f}"
    )
    assert all(np.isfinite(ratio) for ratio in (
        raw_global_ratios
        + raw_mean_ratios
        + converted_global_ratios
        + converted_mean_ratios
    ))
