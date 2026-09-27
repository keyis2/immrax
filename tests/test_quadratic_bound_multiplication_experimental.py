"""Standalone comparison of Taylor and paired-quadratic multiplication.

This test-only experiment represents a paired quadratic bound as ``(lower,
upper)``, where both entries are scalar, zero-remainder ``TaylorModel``
objects over the same normalized source box.  It deliberately does not add a
production domain or change production multiplication.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    TaylorModel,
    evaluate_polynomial,
    polynomial_range,
    taylor_endpoint_models,
    taylor_range,
)

jax.config.update("jax_enable_x64", True)

QuadraticPair = tuple[TaylorModel, TaylorModel]
_TOLERANCE = 2e-12


def _zero_interval(value) -> Interval:
    zero = jnp.zeros_like(jnp.asarray(value))
    return Interval(zero, zero)


def _scalar_model(constant, linear, quadratic, remainder=(0.0, 0.0)):
    return TaylorModel(
        jnp.asarray(constant, dtype=float),
        jnp.asarray(linear, dtype=float),
        jnp.asarray(quadratic, dtype=float),
        Interval(
            jnp.asarray(remainder[0], dtype=float),
            jnp.asarray(remainder[1], dtype=float),
        ),
    )


def _evaluate_pair(pair: QuadraticPair, source):
    lower, upper = pair
    return evaluate_polynomial(lower, source), evaluate_polynomial(upper, source)


def _pair_range(pair: QuadraticPair) -> Interval:
    lower, upper = pair
    return Interval(
        polynomial_range(lower).lower,
        polynomial_range(upper).upper,
    )


def _assert_compatible_pair(pair: QuadraticPair) -> None:
    lower, upper = pair
    if lower.shape != () or upper.shape != ():
        raise ValueError("The experiment supports scalar-valued pairs only.")
    if lower.source_size != upper.source_size:
        raise ValueError("Pair endpoints must use the same source dimension.")
    for endpoint in pair:
        if endpoint.remainder.shape != ():
            raise ValueError("Pair endpoint remainders must be scalar.")
        if not bool(
            jnp.all(endpoint.remainder.lower == 0)
            & jnp.all(endpoint.remainder.upper == 0)
        ):
            raise ValueError("Pair endpoints must have zero remainder.")


def _pair_certificates(pair: QuadraticPair) -> tuple[jax.Array, jax.Array]:
    """Return certified lower bounds for ``upper-lower`` and ``lower``."""

    _assert_compatible_pair(pair)
    lower, upper = pair
    ordering_margin = polynomial_range(upper - lower).lower
    nonnegative_margin = polynomial_range(lower).lower
    return ordering_margin, nonnegative_margin


def _assert_certified_nonnegative_pair(pair: QuadraticPair) -> None:
    ordering_margin, nonnegative_margin = _pair_certificates(pair)
    assert float(ordering_margin) >= -_TOLERANCE
    assert float(nonnegative_margin) >= -_TOLERANCE


def _taylor_to_pair(model: TaylorModel) -> QuadraticPair:
    """Keep the exact source-dependent endpoint polynomials of one tube."""

    if model.shape != ():
        raise ValueError("The experiment supports scalar Taylor models only.")
    return taylor_endpoint_models(model)


def _pair_to_taylor(pair: QuadraticPair) -> TaylorModel:
    """Replace the pair's source-dependent half-width by one global radius."""

    _assert_certified_nonnegative_pair(pair)
    lower, upper = pair
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    half_width_range = polynomial_range(half_width)
    radius = jnp.maximum(
        jnp.abs(half_width_range.lower), jnp.abs(half_width_range.upper)
    )
    return TaylorModel(
        midpoint.constant,
        midpoint.linear,
        midpoint.quadratic,
        Interval(-radius, radius),
    )


def _polynomial_coefficients(model: TaylorModel):
    """Return aggregated ordinary monomial coefficients through degree two."""

    n = model.source_size
    symmetric = 0.5 * (model.quadratic + model.quadratic.T)
    zero = (0,) * n
    coefficients = {zero: model.constant}
    for i in range(n):
        alpha = tuple(1 if axis == i else 0 for axis in range(n))
        coefficients[alpha] = model.linear[i]
    for i in range(n):
        for j in range(i, n):
            alpha = tuple(
                (1 if axis == i else 0) + (1 if axis == j else 0)
                for axis in range(n)
            )
            coefficients[alpha] = (
                0.5 * symmetric[i, i] if i == j else symmetric[i, j]
            )
    return coefficients


def _multiply_coefficients(first: TaylorModel, second: TaylorModel):
    """Multiply and aggregate equal monomials before applying any bound."""

    n = first.source_size
    zero = jnp.zeros_like(first.constant + second.constant)
    product = {}
    for first_alpha, first_coefficient in _polynomial_coefficients(first).items():
        for second_alpha, second_coefficient in _polynomial_coefficients(
            second
        ).items():
            alpha = tuple(
                first_alpha[i] + second_alpha[i] for i in range(n)
            )
            product[alpha] = product.get(alpha, zero) + (
                first_coefficient * second_coefficient
            )
    return product


def _discarded_monomial_correction(alpha, coefficient):
    """Return lower/upper quadratic coefficients enclosing one monomial.

    Put ``t_i = xi_i**2`` and ``w_i = alpha_i / k``.  Weighted AM--GM gives
    ``prod(t_i**w_i) <= sum(w_i*t_i)``.  Since ``k/2 >= 1`` and ``t_i <= 1``,
    ``abs(xi**alpha) = prod(t_i**(alpha_i/2))`` is no larger than that weighted
    geometric mean, proving the requested quadratic envelope.  An all-even
    monomial is nonnegative, so its coefficient sign selects a one-sided
    correction; any odd exponent requires the symmetric envelope.
    """

    k = sum(alpha)
    weights = jnp.asarray(alpha, dtype=jnp.result_type(coefficient, float)) / k
    if all(exponent % 2 == 0 for exponent in alpha):
        lower_scale = jnp.minimum(coefficient, 0.0)
        upper_scale = jnp.maximum(coefficient, 0.0)
    else:
        lower_scale = -jnp.abs(coefficient)
        upper_scale = jnp.abs(coefficient)
    return lower_scale * weights, upper_scale * weights


def _quadratic_from_coefficients(coefficients, source_size, side):
    """Retain degrees 0--2 and add one side of every degree-3/4 envelope."""

    zero_alpha = (0,) * source_size
    template = next(iter(coefficients.values()))
    constant = coefficients.get(zero_alpha, jnp.zeros_like(template))
    linear = jnp.zeros((source_size,), dtype=template.dtype)
    quadratic = jnp.zeros((source_size, source_size), dtype=template.dtype)

    for alpha, coefficient in coefficients.items():
        degree = sum(alpha)
        if degree == 1:
            i = alpha.index(1)
            linear = linear.at[i].add(coefficient)
        elif degree == 2:
            indices = [
                i for i, exponent in enumerate(alpha) for _ in range(exponent)
            ]
            i, j = indices
            if i == j:
                quadratic = quadratic.at[i, i].add(2.0 * coefficient)
            else:
                quadratic = quadratic.at[i, j].add(coefficient)
                quadratic = quadratic.at[j, i].add(coefficient)
        elif degree in (3, 4):
            lower, upper = _discarded_monomial_correction(alpha, coefficient)
            correction = lower if side == "lower" else upper
            diagonal = jnp.diag_indices(source_size)
            quadratic = quadratic.at[diagonal].add(2.0 * correction)
        elif degree != 0:
            raise AssertionError(f"Unexpected product degree {degree}.")

    return TaylorModel(constant, linear, quadratic, _zero_interval(constant))


def _product_envelopes(first: TaylorModel, second: TaylorModel) -> QuadraticPair:
    if first.shape != () or second.shape != ():
        raise ValueError("The experiment supports scalar products only.")
    if first.source_size != second.source_size:
        raise ValueError("Product operands must share one source dimension.")
    coefficients = _multiply_coefficients(first, second)
    return (
        _quadratic_from_coefficients(coefficients, first.source_size, "lower"),
        _quadratic_from_coefficients(coefficients, first.source_size, "upper"),
    )


def _paired_quadratic_product(first: QuadraticPair, second: QuadraticPair):
    """Multiply certified nonnegative pairs using only the needed endpoints."""

    lower_envelope, _ = _product_envelopes(first[0], second[0])
    _, upper_envelope = _product_envelopes(first[1], second[1])
    return lower_envelope, upper_envelope


def _grid(source_size, count=13):
    axis = np.linspace(-1.0, 1.0, count)
    return np.asarray(list(itertools.product(axis, repeat=source_size)))


def _sample_pair(pair: QuadraticPair, sources):
    return np.asarray(
        jax.vmap(lambda source: jnp.stack(_evaluate_pair(pair, source)))(sources)
    )


def _sample_taylor(model: TaylorModel, sources):
    polynomial = np.asarray(
        jax.vmap(lambda source: evaluate_polynomial(model, source))(sources)
    )
    return np.column_stack(
        (
            polynomial + float(model.remainder.lower),
            polynomial + float(model.remainder.upper),
        )
    )


def _assert_finite_model(model: TaylorModel) -> None:
    arrays = (
        model.constant,
        model.linear,
        model.quadratic,
        model.remainder.lower,
        model.remainder.upper,
        polynomial_range(model).lower,
        polynomial_range(model).upper,
        taylor_range(model).lower,
        taylor_range(model).upper,
    )
    assert all(bool(jnp.all(jnp.isfinite(array))) for array in arrays)


def _comparison_metrics(representation, sources, exact_fibers, *, is_pair):
    models = representation if is_pair else (representation,)
    for model in models:
        _assert_finite_model(model)
    assert np.all(np.isfinite(exact_fibers))
    fibers = (
        _sample_pair(representation, sources)
        if is_pair
        else _sample_taylor(representation, sources)
    )
    bounds = _pair_range(representation) if is_pair else taylor_range(representation)
    exact_width = exact_fibers[:, 1] - exact_fibers[:, 0]
    fiber_width = fibers[:, 1] - fibers[:, 0]
    lower_margins = exact_fibers[:, 0] - fibers[:, 0]
    upper_margins = fibers[:, 1] - exact_fibers[:, 1]
    metrics = {
        "lower": float(bounds.lower),
        "upper": float(bounds.upper),
        "width": float(bounds.width),
        "mean_fiber": float(np.mean(fiber_width)),
        "max_fiber": float(np.max(fiber_width)),
        "mean_excess": float(np.mean(fiber_width - exact_width)),
        "max_excess": float(np.max(fiber_width - exact_width)),
        "min_lower_margin": float(np.min(lower_margins)),
        "min_upper_margin": float(np.min(upper_margins)),
    }
    assert np.all(np.isfinite(np.asarray(list(metrics.values()))))
    assert np.all(lower_margins >= -_TOLERANCE)
    assert np.all(upper_margins >= -_TOLERANCE)
    return metrics


def _exact_product_fibers(first: QuadraticPair, second: QuadraticPair, sources):
    first_samples = _sample_pair(first, sources)
    second_samples = _sample_pair(second, sources)
    return np.column_stack(
        (
            first_samples[:, 0] * second_samples[:, 0],
            first_samples[:, 1] * second_samples[:, 1],
        )
    )


def _comparison_a_case():
    first = _scalar_model(
        2.1,
        [0.45, -0.3],
        [[0.32, 0.14], [0.14, 0.18]],
        (-0.17, 0.26),
    )
    second = _scalar_model(
        1.8,
        [-0.34, 0.39],
        [[0.22, -0.12], [-0.12, 0.28]],
        (-0.11, 0.21),
    )
    return first, second


def _comparison_b_case():
    first_midpoint = _scalar_model(
        2.35, [0.32, -0.21], [[0.18, 0.08], [0.08, -0.1]]
    )
    first_half_width = _scalar_model(
        0.08, [0.0, 0.0], [[0.28, 0.0], [0.0, 0.12]]
    )
    second_midpoint = _scalar_model(
        1.95, [-0.24, 0.27], [[-0.08, -0.06], [-0.06, 0.2]]
    )
    second_half_width = _scalar_model(
        0.06, [0.0, 0.0], [[0.16, 0.0], [0.0, 0.24]]
    )
    return (
        (first_midpoint - first_half_width, first_midpoint + first_half_width),
        (second_midpoint - second_half_width, second_midpoint + second_half_width),
    )


def _positive_random_taylor(rng, source_size=2):
    linear = rng.normal(scale=0.16, size=source_size)
    raw_quadratic = rng.normal(scale=0.09, size=(source_size, source_size))
    quadratic = 0.5 * (raw_quadratic + raw_quadratic.T)
    remainder = (-rng.uniform(0.02, 0.12), rng.uniform(0.03, 0.15))
    provisional = _scalar_model(0.0, linear, quadratic, remainder)
    shift = 0.35 - float(taylor_range(provisional).lower)
    return _scalar_model(shift, linear, quadratic, remainder)


def _positive_random_pair(rng, source_size=2):
    midpoint_linear = rng.normal(scale=0.14, size=source_size)
    raw_quadratic = rng.normal(scale=0.07, size=(source_size, source_size))
    midpoint_quadratic = 0.5 * (raw_quadratic + raw_quadratic.T)
    half_width_constant = rng.uniform(0.02, 0.08)
    half_width_weights = rng.uniform(0.02, 0.12, size=source_size)
    half_width = _scalar_model(
        half_width_constant,
        np.zeros(source_size),
        np.diag(2.0 * half_width_weights),
    )
    provisional_midpoint = _scalar_model(
        0.0, midpoint_linear, midpoint_quadratic
    )
    provisional_lower = provisional_midpoint - half_width
    shift = 0.35 - float(polynomial_range(provisional_lower).lower)
    midpoint = _scalar_model(shift, midpoint_linear, midpoint_quadratic)
    return midpoint - half_width, midpoint + half_width


def _outcome(candidate, baseline, key, tolerance=1e-11):
    difference = candidate[key] - baseline[key]
    if difference < -tolerance:
        return "win"
    if difference > tolerance:
        return "loss"
    return "tie"


def _format_metrics(title, rows):
    headings = (
        "method",
        "global [lower, upper] / width",
        "fiber mean / max",
        "excess mean / max",
        "min margins L / U",
    )
    lines = [title, " | ".join(headings)]
    for name, metrics in rows:
        lines.append(
            f"{name} | [{metrics['lower']:.6f}, {metrics['upper']:.6f}] / "
            f"{metrics['width']:.6f} | {metrics['mean_fiber']:.6f} / "
            f"{metrics['max_fiber']:.6f} | {metrics['mean_excess']:.6f} / "
            f"{metrics['max_excess']:.6f} | {metrics['min_lower_margin']:.3e} / "
            f"{metrics['min_upper_margin']:.3e}"
        )
    return "\n".join(lines)


def test_weighted_am_gm_monomial_envelopes():
    sources = _grid(3, count=9)
    cases = [
        ((3, 0, 0), 0.7),
        ((2, 1, 0), -0.6),
        ((1, 1, 2), 0.5),
        ((2, 2, 0), 0.8),
        ((2, 0, 2), -0.9),
    ]
    for alpha, coefficient in cases:
        lower_coefficients, upper_coefficients = _discarded_monomial_correction(
            alpha, jnp.asarray(coefficient)
        )
        exact = coefficient * np.prod(
            sources ** np.asarray(alpha)[None, :], axis=1
        )
        lower = np.sum(np.asarray(lower_coefficients) * sources**2, axis=1)
        upper = np.sum(np.asarray(upper_coefficients) * sources**2, axis=1)
        assert np.all(exact >= lower - _TOLERANCE)
        assert np.all(exact <= upper + _TOLERANCE)


def test_affine_times_affine_has_no_discarded_terms():
    first = _scalar_model(1.4, [0.2, -0.3], np.zeros((2, 2)))
    second = _scalar_model(1.1, [-0.1, 0.25], np.zeros((2, 2)))
    lower, upper = _product_envelopes(first, second)
    np.testing.assert_allclose(lower.constant, upper.constant)
    np.testing.assert_allclose(lower.linear, upper.linear)
    np.testing.assert_allclose(lower.quadratic, upper.quadratic)
    np.testing.assert_allclose(lower.remainder.width, 0.0)
    np.testing.assert_allclose(upper.remainder.width, 0.0)
    sources = _grid(2, count=11)
    exact = np.asarray(
        jax.vmap(
            lambda source: evaluate_polynomial(first, source)
            * evaluate_polynomial(second, source)
        )(sources)
    )
    np.testing.assert_allclose(_sample_pair((lower, upper), sources)[:, 0], exact)
    np.testing.assert_allclose(_sample_pair((lower, upper), sources)[:, 1], exact)


def test_exact_conversions_and_pair_to_taylor_sampled_containment():
    first, _ = _comparison_a_case()
    pair = _taylor_to_pair(first)
    np.testing.assert_allclose(pair[0].constant, first.constant + first.remainder.lower)
    np.testing.assert_allclose(pair[1].constant, first.constant + first.remainder.upper)
    for endpoint in pair:
        np.testing.assert_allclose(endpoint.linear, first.linear)
        np.testing.assert_allclose(endpoint.quadratic, first.quadratic)
        np.testing.assert_allclose(endpoint.remainder.width, 0.0)
    sources = _grid(first.source_size, count=15)
    pair_samples = _sample_pair(pair, sources)
    polynomial = np.asarray(
        jax.vmap(lambda source: evaluate_polynomial(first, source))(sources)
    )
    np.testing.assert_allclose(
        pair_samples[:, 0], polynomial + float(first.remainder.lower)
    )
    np.testing.assert_allclose(
        pair_samples[:, 1], polynomial + float(first.remainder.upper)
    )

    variable_width_pair = _comparison_b_case()[0]
    converted = _pair_to_taylor(variable_width_pair)
    midpoint = 0.5 * (variable_width_pair[0] + variable_width_pair[1])
    half_width = 0.5 * (variable_width_pair[1] - variable_width_pair[0])
    half_width_range = polynomial_range(half_width)
    expected_radius = max(
        abs(float(half_width_range.lower)), abs(float(half_width_range.upper))
    )
    np.testing.assert_allclose(converted.constant, midpoint.constant)
    np.testing.assert_allclose(converted.linear, midpoint.linear)
    np.testing.assert_allclose(converted.quadratic, midpoint.quadratic)
    np.testing.assert_allclose(converted.remainder.lower, -expected_radius)
    np.testing.assert_allclose(converted.remainder.upper, expected_radius)
    original = _sample_pair(variable_width_pair, sources)
    converted_samples = _sample_taylor(converted, sources)
    assert np.all(converted_samples[:, 0] <= original[:, 0] + _TOLERANCE)
    assert np.all(converted_samples[:, 1] >= original[:, 1] - _TOLERANCE)


def test_fixed_shape_paired_product_jit_smoke():
    first, second = _comparison_b_case()
    for pair in (first, second):
        _assert_certified_nonnegative_pair(pair)
    eager = _paired_quadratic_product(first, second)
    compiled = jax.jit(_paired_quadratic_product)(first, second)
    for eager_endpoint, compiled_endpoint in zip(eager, compiled):
        np.testing.assert_allclose(compiled_endpoint.constant, eager_endpoint.constant)
        np.testing.assert_allclose(compiled_endpoint.linear, eager_endpoint.linear)
        np.testing.assert_allclose(
            compiled_endpoint.quadratic, eager_endpoint.quadratic
        )


def test_deterministic_and_random_comparisons():
    sources = _grid(2, count=17)

    first_taylor, second_taylor = _comparison_a_case()
    first_pair = _taylor_to_pair(first_taylor)
    second_pair = _taylor_to_pair(second_taylor)
    for pair in (first_pair, second_pair):
        _assert_certified_nonnegative_pair(pair)
    exact_a = _exact_product_fibers(first_pair, second_pair, sources)
    existing_a = _comparison_metrics(
        first_taylor * second_taylor, sources, exact_a, is_pair=False
    )
    paired_a = _comparison_metrics(
        _paired_quadratic_product(first_pair, second_pair),
        sources,
        exact_a,
        is_pair=True,
    )
    # The pair keeps each asymmetric endpoint's source dependence instead of
    # combining truncation and remainder interactions in one global interval.
    assert paired_a["width"] < existing_a["width"] - 1e-6
    assert paired_a["mean_fiber"] < existing_a["mean_fiber"] - 1e-6

    first_bound, second_bound = _comparison_b_case()
    for pair in (first_bound, second_bound):
        _assert_certified_nonnegative_pair(pair)
    exact_b = _exact_product_fibers(first_bound, second_bound, sources)
    paired_b = _comparison_metrics(
        _paired_quadratic_product(first_bound, second_bound),
        sources,
        exact_b,
        is_pair=True,
    )
    converted_b = _comparison_metrics(
        _pair_to_taylor(first_bound) * _pair_to_taylor(second_bound),
        sources,
        exact_b,
        is_pair=False,
    )
    # Conversion replaces D(xi) by max(D), widening sources where D is smaller;
    # the direct pair retains that varying fiber width through multiplication.
    assert paired_b["width"] < converted_b["width"] - 1e-6
    assert paired_b["mean_fiber"] < converted_b["mean_fiber"] - 1e-6

    rng = np.random.default_rng(20260926)
    counts = {
        "A global": {"win": 0, "loss": 0, "tie": 0},
        "A mean": {"win": 0, "loss": 0, "tie": 0},
        "B global": {"win": 0, "loss": 0, "tie": 0},
        "B mean": {"win": 0, "loss": 0, "tie": 0},
    }
    random_minimum_margin = np.inf
    for _ in range(12):
        random_first_taylor = _positive_random_taylor(rng)
        random_second_taylor = _positive_random_taylor(rng)
        random_first_pair = _taylor_to_pair(random_first_taylor)
        random_second_pair = _taylor_to_pair(random_second_taylor)
        for pair in (random_first_pair, random_second_pair):
            _assert_certified_nonnegative_pair(pair)
        exact = _exact_product_fibers(
            random_first_pair, random_second_pair, sources
        )
        existing = _comparison_metrics(
            random_first_taylor * random_second_taylor,
            sources,
            exact,
            is_pair=False,
        )
        paired = _comparison_metrics(
            _paired_quadratic_product(random_first_pair, random_second_pair),
            sources,
            exact,
            is_pair=True,
        )
        counts["A global"][_outcome(paired, existing, "width")] += 1
        counts["A mean"][_outcome(paired, existing, "mean_fiber")] += 1
        random_minimum_margin = min(
            random_minimum_margin,
            paired["min_lower_margin"],
            paired["min_upper_margin"],
            existing["min_lower_margin"],
            existing["min_upper_margin"],
        )

        random_first_bound = _positive_random_pair(rng)
        random_second_bound = _positive_random_pair(rng)
        for pair in (random_first_bound, random_second_bound):
            _assert_certified_nonnegative_pair(pair)
        exact = _exact_product_fibers(
            random_first_bound, random_second_bound, sources
        )
        paired = _comparison_metrics(
            _paired_quadratic_product(random_first_bound, random_second_bound),
            sources,
            exact,
            is_pair=True,
        )
        converted = _comparison_metrics(
            _pair_to_taylor(random_first_bound)
            * _pair_to_taylor(random_second_bound),
            sources,
            exact,
            is_pair=False,
        )
        counts["B global"][_outcome(paired, converted, "width")] += 1
        counts["B mean"][_outcome(paired, converted, "mean_fiber")] += 1
        random_minimum_margin = min(
            random_minimum_margin,
            paired["min_lower_margin"],
            paired["min_upper_margin"],
            converted["min_lower_margin"],
            converted["min_upper_margin"],
        )

    summary = [
        _format_metrics(
            "Comparison A: exact Taylor endpoints before multiplication",
            [("existing Taylor", existing_a), ("paired quadratic", paired_a)],
        ),
        _format_metrics(
            "Comparison B: source-dependent quadratic input fibers",
            [("direct paired", paired_b), ("converted Taylor", converted_b)],
        ),
        "Random paired-vs-Taylor outcomes (win/loss/tie):",
    ]
    for label, result in counts.items():
        summary.append(
            f"{label}: {result['win']}/{result['loss']}/{result['tie']}"
        )
    summary.extend(
        (
            f"Minimum sampled containment margin: {random_minimum_margin:.3e}",
            "Sampling checks implementation regression and tightness only; "
            "the coefficientwise monomial envelopes provide analytic soundness.",
            "No production behavior changed.",
        )
    )
    print("\n\n" + "\n".join(summary))
