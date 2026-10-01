"""Compare O(n^2) shared-quadratic multiplication with Taylor multiplication.

The test-local bound

    lower(xi) = c_l + a_l @ xi + 0.5 * xi.T @ Q @ xi
    upper(xi) = c_u + a_u @ xi + 0.5 * xi.T @ Q @ xi

stores one common quadratic matrix and an affine source-dependent fiber width.
Its main general-sign multiplication keeps Taylor's degree-2 core and uses a
fixed number of centered product identities for the omitted terms, so it never
constructs cubic or quartic coefficient tensors.  Sampling below checks
implementation containment and tightness; soundness follows from the analytic
residual bounds documented on ``_retained_core_product``.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    QuadraticEndpointPair,
    TaylorModel,
    evaluate_polynomial,
    nonnegative_endpoint_product,
    polynomial_range,
    taylor_endpoint_models,
    taylor_range,
)

jax.config.update("jax_enable_x64", True)

_TOLERANCE = 2e-12


class SharedQuadraticBound(NamedTuple):
    """Scalar test-local quadratic backbone with two affine endpoints."""

    quadratic: jax.Array
    lower_linear: jax.Array
    lower_constant: jax.Array
    upper_linear: jax.Array
    upper_constant: jax.Array


def _zero_interval(value) -> Interval:
    zero = jnp.zeros_like(jnp.asarray(value))
    return Interval(zero, zero)


def _model(constant, linear, quadratic, remainder=(0.0, 0.0)):
    return TaylorModel(
        jnp.asarray(constant, dtype=float),
        jnp.asarray(linear, dtype=float),
        jnp.asarray(quadratic, dtype=float),
        Interval(
            jnp.asarray(remainder[0], dtype=float),
            jnp.asarray(remainder[1], dtype=float),
        ),
    )


def _endpoint_models(bound: SharedQuadraticBound):
    return (
        TaylorModel(
            bound.lower_constant,
            bound.lower_linear,
            bound.quadratic,
            _zero_interval(bound.lower_constant),
        ),
        TaylorModel(
            bound.upper_constant,
            bound.upper_linear,
            bound.quadratic,
            _zero_interval(bound.upper_constant),
        ),
    )


def _bound_range(bound: SharedQuadraticBound) -> Interval:
    lower, upper = _endpoint_models(bound)
    return Interval(
        polynomial_range(lower).lower,
        polynomial_range(upper).upper,
    )


def _ordering_margin(bound: SharedQuadraticBound):
    difference = _model(
        bound.upper_constant - bound.lower_constant,
        bound.upper_linear - bound.lower_linear,
        jnp.zeros_like(bound.quadratic),
    )
    return polynomial_range(difference).lower


def _assert_valid(bound: SharedQuadraticBound) -> None:
    n = bound.lower_linear.shape[-1]
    assert bound.quadratic.shape == (n, n)
    assert bound.upper_linear.shape == (n,)
    assert bound.lower_constant.shape == ()
    assert bound.upper_constant.shape == ()
    assert float(_ordering_margin(bound)) >= -_TOLERANCE
    arrays = (*bound, _bound_range(bound).lower, _bound_range(bound).upper)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in arrays)


def _scale_bound(bound: SharedQuadraticBound, scale):
    """Exactly scale one bound by a point scalar, swapping affine endpoints."""

    scale = jnp.asarray(scale)
    nonnegative = scale >= 0
    return SharedQuadraticBound(
        scale * bound.quadratic,
        jnp.where(
            nonnegative,
            scale * bound.lower_linear,
            scale * bound.upper_linear,
        ),
        jnp.where(
            nonnegative,
            scale * bound.lower_constant,
            scale * bound.upper_constant,
        ),
        jnp.where(
            nonnegative,
            scale * bound.upper_linear,
            scale * bound.lower_linear,
        ),
        jnp.where(
            nonnegative,
            scale * bound.upper_constant,
            scale * bound.lower_constant,
        ),
    )


def _add_bounds(first: SharedQuadraticBound, second: SharedQuadraticBound):
    return SharedQuadraticBound(
        first.quadratic + second.quadratic,
        first.lower_linear + second.lower_linear,
        first.lower_constant + second.lower_constant,
        first.upper_linear + second.upper_linear,
        first.upper_constant + second.upper_constant,
    )


def _centered_product(
    first: SharedQuadraticBound, second: SharedQuadraticBound
) -> SharedQuadraticBound:
    """Apply an O(n^2), general-sign centered McCormick relaxation.

    If global ranges are ``x in [p_l, p_u]`` and ``y in [q_l, q_u]``, set
    their centers to ``p_c, q_c`` and radii to ``p_r, q_r``.  The identity

        x*y = q_c*x + p_c*y - p_c*q_c
              + (x-p_c)*(y-q_c)

    and ``(x-p_c)*(y-q_c) in [-p_r*q_r, p_r*q_r]`` are valid for arbitrary
    signs and sign crossings.  Constant scaling is endpoint-sign-aware.  Both
    output endpoints therefore share ``q_c*Q_x + p_c*Q_y``.  The rule uses
    dense quadratic scaling/ranging only and is O(n^2), like Taylor ``_mul``.
    """

    first_range = _bound_range(first)
    second_range = _bound_range(second)
    first_center = 0.5 * (first_range.lower + first_range.upper)
    second_center = 0.5 * (second_range.lower + second_range.upper)
    first_radius = 0.5 * (first_range.upper - first_range.lower)
    second_radius = 0.5 * (second_range.upper - second_range.lower)

    retained = _add_bounds(
        _scale_bound(first, second_center),
        _scale_bound(second, first_center),
    )
    common_shift = -first_center * second_center
    residual_radius = first_radius * second_radius
    return SharedQuadraticBound(
        retained.quadratic,
        retained.lower_linear,
        retained.lower_constant + common_shift - residual_radius,
        retained.upper_linear,
        retained.upper_constant + common_shift + residual_radius,
    )


def _expansion_product(
    first: SharedQuadraticBound,
    second: SharedQuadraticBound,
    first_expansion,
    second_expansion,
) -> SharedQuadraticBound:
    """Apply one sound general-sign shifted-product relaxation.

    For fixed expansion points ``alpha`` and ``beta``, use

        x*y = beta*x + alpha*y - alpha*beta
              + (x-alpha)*(y-beta).

    The final product is bounded by ordinary four-corner interval arithmetic
    on the two shifted global ranges.  Only this residual is intervalized;
    the source dependence of ``beta*x + alpha*y`` remains in the result.
    Both endpoints share ``beta*Q_x + alpha*Q_y``.  Each fixed candidate is
    O(n^2), and a fixed-size portfolio remains O(n^2).
    """

    first_range = _bound_range(first)
    second_range = _bound_range(second)
    first_shifted = (
        first_range.lower - first_expansion,
        first_range.upper - first_expansion,
    )
    second_shifted = (
        second_range.lower - second_expansion,
        second_range.upper - second_expansion,
    )
    residual_products = jnp.stack(
        (
            first_shifted[0] * second_shifted[0],
            first_shifted[0] * second_shifted[1],
            first_shifted[1] * second_shifted[0],
            first_shifted[1] * second_shifted[1],
        )
    )
    residual_lower = jnp.min(residual_products)
    residual_upper = jnp.max(residual_products)
    retained = _add_bounds(
        _scale_bound(first, second_expansion),
        _scale_bound(second, first_expansion),
    )
    common_shift = -first_expansion * second_expansion
    return retained._replace(
        lower_constant=(
            retained.lower_constant + common_shift + residual_lower
        ),
        upper_constant=(
            retained.upper_constant + common_shift + residual_upper
        ),
    )


def _portfolio_product(
    first: SharedQuadraticBound, second: SharedQuadraticBound
) -> SharedQuadraticBound:
    """Select the narrowest global bound from nine O(n^2) expansions."""

    first_range = _bound_range(first)
    second_range = _bound_range(second)
    first_points = (
        first_range.lower,
        first_range.center,
        first_range.upper,
    )
    second_points = (
        second_range.lower,
        second_range.center,
        second_range.upper,
    )
    candidates = tuple(
        _expansion_product(first, second, first_point, second_point)
        for first_point in first_points
        for second_point in second_points
    )
    widths = jnp.stack([_bound_range(candidate).width for candidate in candidates])
    selected = jnp.argmin(widths)
    stacked = jax.tree.map(lambda *values: jnp.stack(values), *candidates)
    return jax.tree.map(lambda values: values[selected], stacked)


def _midpoint_and_half_width(bound: SharedQuadraticBound):
    midpoint = _model(
        0.5 * (bound.lower_constant + bound.upper_constant),
        0.5 * (bound.lower_linear + bound.upper_linear),
        bound.quadratic,
    )
    half_width = _model(
        0.5 * (bound.upper_constant - bound.lower_constant),
        0.5 * (bound.upper_linear - bound.lower_linear),
        jnp.zeros_like(bound.quadratic),
    )
    return midpoint, half_width


def _retained_degree_two_product(first: TaylorModel, second: TaylorModel):
    """Return exactly the degree-zero, -one, and -two product terms."""

    constant = first.constant * second.constant
    linear = (
        first.linear * second.constant
        + second.linear * first.constant
    )
    quadratic = (
        first.quadratic * second.constant
        + second.quadratic * first.constant
        + first.linear[:, None] * second.linear[None, :]
        + second.linear[:, None] * first.linear[None, :]
    )
    return TaylorModel(
        constant,
        linear,
        quadratic,
        _zero_interval(constant),
    )


def _linear_component(model: TaylorModel):
    return _model(
        0.0,
        model.linear,
        jnp.zeros_like(model.quadratic),
    )


def _quadratic_component(model: TaylorModel):
    return _model(
        0.0,
        jnp.zeros_like(model.linear),
        model.quadratic,
    )


def _centered_polynomial_product(first: TaylorModel, second: TaylorModel):
    """Return a degree-at-most-two center and a constant residual radius."""

    first_range = polynomial_range(first)
    second_range = polynomial_range(second)
    first_center = first_range.center
    second_center = second_range.center
    first_radius = first_range.pert
    second_radius = second_range.pert
    approximation = (
        second_center * first
        + first_center * second
        - first_center * second_center
    )
    return approximation, first_radius * second_radius


def _retained_core_product(
    first: SharedQuadraticBound, second: SharedQuadraticBound
) -> SharedQuadraticBound:
    """Keep Taylor's exact degree-2 core and give its error affine width.

    Write each midpoint polynomial as ``M = c + l + q``.  The exact retained
    Taylor product is kept through degree two.  Each omitted product
    ``l_x*q_y``, ``q_x*l_y``, and ``q_x*q_y`` is relaxed with the centered
    identity while retaining its affine/quadratic center; only its centered
    residual becomes a constant radius.

    For affine nonnegative half-widths ``D_x`` and ``D_y``, the uncertain
    interactions are bounded by

        abs(M_x) D_y + abs(M_y) D_x + D_x D_y.

    Global polynomial magnitudes multiply the opposite affine width, while a
    centered affine upper plane bounds ``D_x D_y``.  The resulting half-width
    is affine in the shared sources.  All ranges and dense matrix operations
    are O(n^2); no cubic or quartic coefficient tensor is formed.
    """

    first_midpoint, first_width = _midpoint_and_half_width(first)
    second_midpoint, second_width = _midpoint_and_half_width(second)
    retained = _retained_degree_two_product(first_midpoint, second_midpoint)

    first_linear = _linear_component(first_midpoint)
    second_linear = _linear_component(second_midpoint)
    first_quadratic = _quadratic_component(first_midpoint)
    second_quadratic = _quadratic_component(second_midpoint)
    overflow_terms = (
        _centered_polynomial_product(first_linear, second_quadratic),
        _centered_polynomial_product(first_quadratic, second_linear),
        _centered_polynomial_product(first_quadratic, second_quadratic),
    )
    common = retained
    overflow_radius = jnp.zeros_like(retained.constant)
    for approximation, radius in overflow_terms:
        common = common + approximation
        overflow_radius = overflow_radius + radius

    first_midpoint_range = polynomial_range(first_midpoint)
    second_midpoint_range = polynomial_range(second_midpoint)
    first_magnitude = jnp.maximum(
        jnp.abs(first_midpoint_range.lower),
        jnp.abs(first_midpoint_range.upper),
    )
    second_magnitude = jnp.maximum(
        jnp.abs(second_midpoint_range.lower),
        jnp.abs(second_midpoint_range.upper),
    )
    width_product_center, width_product_radius = _centered_polynomial_product(
        first_width, second_width
    )
    width_product_upper_constant = (
        width_product_center.constant + width_product_radius
    )
    output_width_constant = (
        overflow_radius
        + first_magnitude * second_width.constant
        + second_magnitude * first_width.constant
        + width_product_upper_constant
    )
    output_width_linear = (
        first_magnitude * second_width.linear
        + second_magnitude * first_width.linear
        + width_product_center.linear
    )
    return SharedQuadraticBound(
        common.quadratic,
        common.linear - output_width_linear,
        common.constant - output_width_constant,
        common.linear + output_width_linear,
        common.constant + output_width_constant,
    )


def _bound_to_taylor(bound: SharedQuadraticBound) -> TaylorModel:
    """Concretize only the affine half-width into a Taylor remainder."""

    midpoint_constant = 0.5 * (bound.lower_constant + bound.upper_constant)
    midpoint_linear = 0.5 * (bound.lower_linear + bound.upper_linear)
    half_width = _model(
        0.5 * (bound.upper_constant - bound.lower_constant),
        0.5 * (bound.upper_linear - bound.lower_linear),
        jnp.zeros_like(bound.quadratic),
    )
    half_width_range = polynomial_range(half_width)
    radius = jnp.maximum(
        jnp.abs(half_width_range.lower), jnp.abs(half_width_range.upper)
    )
    return TaylorModel(
        midpoint_constant,
        midpoint_linear,
        bound.quadratic,
        Interval(-radius, radius),
    )


def _symmetric_bound(midpoint: TaylorModel, half_width_constant, half_width_linear):
    half_width_constant = jnp.asarray(half_width_constant, dtype=float)
    half_width_linear = jnp.asarray(half_width_linear, dtype=float)
    return SharedQuadraticBound(
        midpoint.quadratic,
        midpoint.linear - half_width_linear,
        midpoint.constant - half_width_constant,
        midpoint.linear + half_width_linear,
        midpoint.constant + half_width_constant,
    )


def _shift_bound(bound: SharedQuadraticBound, shift):
    return bound._replace(
        lower_constant=bound.lower_constant + shift,
        upper_constant=bound.upper_constant + shift,
    )


def _set_range_center_ratio(bound: SharedQuadraticBound, ratio):
    bounds = _bound_range(bound)
    center = 0.5 * (bounds.lower + bounds.upper)
    radius = 0.5 * (bounds.upper - bounds.lower)
    return _shift_bound(bound, ratio * radius - center)


def _evaluate_bound(bound: SharedQuadraticBound, sources):
    lower, upper = _endpoint_models(bound)
    return np.column_stack(
        (
            np.asarray(jax.vmap(lambda z: evaluate_polynomial(lower, z))(sources)),
            np.asarray(jax.vmap(lambda z: evaluate_polynomial(upper, z))(sources)),
        )
    )


def _evaluate_taylor(model: TaylorModel, sources):
    polynomial = np.asarray(
        jax.vmap(lambda z: evaluate_polynomial(model, z))(sources)
    )
    return np.column_stack(
        (
            polynomial + float(model.remainder.lower),
            polynomial + float(model.remainder.upper),
        )
    )


def _exact_product_fibers(
    first: SharedQuadraticBound, second: SharedQuadraticBound, sources
):
    first_fibers = _evaluate_bound(first, sources)
    second_fibers = _evaluate_bound(second, sources)
    products = np.stack(
        (
            first_fibers[:, 0] * second_fibers[:, 0],
            first_fibers[:, 0] * second_fibers[:, 1],
            first_fibers[:, 1] * second_fibers[:, 0],
            first_fibers[:, 1] * second_fibers[:, 1],
        ),
        axis=1,
    )
    return np.column_stack((np.min(products, axis=1), np.max(products, axis=1)))


def _metrics(representation, sources, exact_fibers, *, shared):
    if shared:
        fibers = _evaluate_bound(representation, sources)
        bounds = _bound_range(representation)
    else:
        fibers = _evaluate_taylor(representation, sources)
        bounds = taylor_range(representation)
    exact_width = exact_fibers[:, 1] - exact_fibers[:, 0]
    fiber_width = fibers[:, 1] - fibers[:, 0]
    lower_margin = exact_fibers[:, 0] - fibers[:, 0]
    upper_margin = fibers[:, 1] - exact_fibers[:, 1]
    result = {
        "lower": float(bounds.lower),
        "upper": float(bounds.upper),
        "global_width": float(bounds.width),
        "mean_width": float(np.mean(fiber_width)),
        "min_width": float(np.min(fiber_width)),
        "max_width": float(np.max(fiber_width)),
        "mean_excess": float(np.mean(fiber_width - exact_width)),
        "max_excess": float(np.max(fiber_width - exact_width)),
        "min_lower_margin": float(np.min(lower_margin)),
        "min_upper_margin": float(np.min(upper_margin)),
    }
    assert np.all(np.isfinite(np.asarray(list(result.values()))))
    assert np.all(lower_margin >= -_TOLERANCE)
    assert np.all(upper_margin >= -_TOLERANCE)
    return result


def _grid(source_size, count=17):
    axis = np.linspace(-1.0, 1.0, count)
    return np.asarray(list(itertools.product(axis, repeat=source_size)))


def _deterministic_case():
    first_midpoint = _model(
        0.75,
        [0.65, -0.25],
        [[0.18, 0.08], [0.08, -0.10]],
    )
    second_midpoint = _model(
        0.55,
        [-0.45, 0.50],
        [[-0.12, -0.06], [-0.06, 0.16]],
    )
    first = _symmetric_bound(first_midpoint, 0.18, [0.06, -0.04])
    second = _symmetric_bound(second_midpoint, 0.16, [-0.05, 0.05])
    return first, second


def _random_bound(rng, source_size, center_ratio):
    linear = rng.normal(scale=0.32, size=source_size)
    raw_quadratic = rng.normal(scale=0.08, size=(source_size, source_size))
    quadratic = 0.5 * (raw_quadratic + raw_quadratic.T)
    half_width_linear = rng.normal(scale=0.035, size=source_size)
    half_width_constant = np.sum(np.abs(half_width_linear)) + rng.uniform(
        0.025, 0.09
    )
    provisional = _symmetric_bound(
        _model(0.0, linear, quadratic),
        half_width_constant,
        half_width_linear,
    )
    return _set_range_center_ratio(provisional, center_ratio)


def _outcome(candidate, baseline, key, tolerance=1e-11):
    difference = candidate[key] - baseline[key]
    if difference < -tolerance:
        return "win"
    if difference > tolerance:
        return "loss"
    return "tie"


def _format_metrics(title, rows):
    lines = [
        title,
        "method | global width | fiber min/mean/max | excess mean/max | "
        "min margins L/U",
    ]
    for name, values in rows:
        lines.append(
            f"{name} | {values['global_width']:.6f} | "
            f"{values['min_width']:.6f}/{values['mean_width']:.6f}/"
            f"{values['max_width']:.6f} | {values['mean_excess']:.6f}/"
            f"{values['max_excess']:.6f} | {values['min_lower_margin']:.3e}/"
            f"{values['min_upper_margin']:.3e}"
        )
    return "\n".join(lines)


def test_centered_general_sign_rule_is_jittable_and_preserves_shared_quadratic():
    first, second = _deterministic_case()
    for bound in (first, second):
        _assert_valid(bound)
        bounds = _bound_range(bound)
        assert float(bounds.lower) < 0.0 < float(bounds.upper)

    eager = _centered_product(first, second)
    compiled = jax.jit(_centered_product)(first, second)
    expected_quadratic = (
        _bound_range(second).center * first.quadratic
        + _bound_range(first).center * second.quadratic
    )
    np.testing.assert_allclose(eager.quadratic, expected_quadratic)
    for eager_leaf, compiled_leaf in zip(eager, compiled):
        np.testing.assert_allclose(compiled_leaf, eager_leaf)
    _assert_valid(compiled)

    eager_portfolio = _portfolio_product(first, second)
    compiled_portfolio = jax.jit(_portfolio_product)(first, second)
    for eager_leaf, compiled_leaf in zip(eager_portfolio, compiled_portfolio):
        np.testing.assert_allclose(compiled_leaf, eager_leaf)
    _assert_valid(compiled_portfolio)

    eager_retained = _retained_core_product(first, second)
    compiled_retained = jax.jit(_retained_core_product)(first, second)
    for eager_leaf, compiled_leaf in zip(eager_retained, compiled_retained):
        np.testing.assert_allclose(compiled_leaf, eager_leaf)
    _assert_valid(compiled_retained)


def test_affine_width_capture_against_taylor_remainder():
    sources = _grid(2, count=17)
    first, second = _deterministic_case()
    for bound in (first, second):
        _assert_valid(bound)

    exact = _exact_product_fibers(first, second, sources)
    centered_product = _centered_product(first, second)
    portfolio_product = _portfolio_product(first, second)
    shared_product = _retained_core_product(first, second)
    shared_concretized = _bound_to_taylor(shared_product)
    taylor_product = _bound_to_taylor(first) * _bound_to_taylor(second)
    shared_metrics = _metrics(shared_product, sources, exact, shared=True)
    centered_metrics = _metrics(centered_product, sources, exact, shared=True)
    portfolio_metrics = _metrics(portfolio_product, sources, exact, shared=True)
    concretized_metrics = _metrics(
        shared_concretized, sources, exact, shared=False
    )
    taylor_metrics = _metrics(taylor_product, sources, exact, shared=False)

    # This strict comparison isolates only the value of retaining the affine
    # output width: both rows use the identical centered multiplication result.
    assert shared_metrics["mean_width"] < concretized_metrics["mean_width"] - 1e-8
    assert shared_metrics["max_width"] <= concretized_metrics["max_width"] + 1e-12

    rng = np.random.default_rng(20260927)
    samples = rng.uniform(-1.0, 1.0, size=(257, 3))
    counts = {
        "shared vs native Taylor global": {"win": 0, "loss": 0, "tie": 0},
        "shared vs native Taylor mean": {"win": 0, "loss": 0, "tie": 0},
        "affine vs concretized global": {"win": 0, "loss": 0, "tie": 0},
        "affine vs concretized mean": {"win": 0, "loss": 0, "tie": 0},
    }
    minimum_margin = np.inf
    affine_mean_reductions = []
    native_global_reductions = []
    native_mean_reductions = []
    concretized_native_mean_reductions = []
    native_mean_by_category = {"crossing": [], "fixed-sign": []}
    crossing_cases = 0
    center_ratios = (0.40, 1.25, -1.25, 0.65)
    for case in range(20):
        random_first = _random_bound(rng, 3, center_ratios[case % 4])
        random_second = _random_bound(rng, 3, center_ratios[(case + 1) % 4])
        input_crossings = []
        for bound in (random_first, random_second):
            _assert_valid(bound)
            bounds = _bound_range(bound)
            crosses = float(bounds.lower) < 0.0 < float(bounds.upper)
            input_crossings.append(crosses)
            crossing_cases += int(crosses)

        exact_random = _exact_product_fibers(
            random_first, random_second, samples
        )
        shared_random = _retained_core_product(random_first, random_second)
        concretized_random = _bound_to_taylor(shared_random)
        taylor_random = _bound_to_taylor(random_first) * _bound_to_taylor(
            random_second
        )
        shared_result = _metrics(
            shared_random, samples, exact_random, shared=True
        )
        concretized_result = _metrics(
            concretized_random, samples, exact_random, shared=False
        )
        taylor_result = _metrics(
            taylor_random, samples, exact_random, shared=False
        )
        counts["shared vs native Taylor global"][
            _outcome(shared_result, taylor_result, "global_width")
        ] += 1
        counts["shared vs native Taylor mean"][
            _outcome(shared_result, taylor_result, "mean_width")
        ] += 1
        counts["affine vs concretized global"][
            _outcome(shared_result, concretized_result, "global_width")
        ] += 1
        counts["affine vs concretized mean"][
            _outcome(shared_result, concretized_result, "mean_width")
        ] += 1
        affine_mean_reductions.append(
            1.0 - shared_result["mean_width"] / concretized_result["mean_width"]
        )
        native_global_reductions.append(
            1.0
            - shared_result["global_width"] / taylor_result["global_width"]
        )
        native_mean_reduction = (
            1.0 - shared_result["mean_width"] / taylor_result["mean_width"]
        )
        native_mean_reductions.append(native_mean_reduction)
        concretized_native_mean_reductions.append(
            1.0
            - concretized_result["mean_width"] / taylor_result["mean_width"]
        )
        category = "crossing" if any(input_crossings) else "fixed-sign"
        native_mean_by_category[category].append(native_mean_reduction)
        minimum_margin = min(
            minimum_margin,
            shared_result["min_lower_margin"],
            shared_result["min_upper_margin"],
            concretized_result["min_lower_margin"],
            concretized_result["min_upper_margin"],
            taylor_result["min_lower_margin"],
            taylor_result["min_upper_margin"],
        )

    deterministic_reduction = (
        1.0
        - shared_metrics["mean_width"] / concretized_metrics["mean_width"]
    )
    deterministic_native_reduction = (
        1.0 - shared_metrics["mean_width"] / taylor_metrics["mean_width"]
    )
    summary = [
        _format_metrics(
            "Deterministic two-source general-sign comparison",
            (
                ("retained-core affine width", shared_metrics),
                ("shifted-product portfolio", portfolio_metrics),
                ("centered shared-Q", centered_metrics),
                ("same result, width concretized", concretized_metrics),
                ("native Taylor multiplication", taylor_metrics),
            ),
        ),
        f"Deterministic affine-width mean reduction: "
        f"{100.0 * deterministic_reduction:.2f}%",
        f"Deterministic shared-vs-native-Taylor mean reduction: "
        f"{100.0 * deterministic_native_reduction:.2f}%",
        "Random outcomes (win/loss/tie):",
    ]
    for label, result in counts.items():
        summary.append(
            f"{label}: {result['win']}/{result['loss']}/{result['tie']}"
        )
    summary.extend(
        (
            f"Random mean affine-width reduction: "
            f"{100.0 * np.mean(affine_mean_reductions):.2f}% "
            f"(min {100.0 * np.min(affine_mean_reductions):.2f}%, "
            f"max {100.0 * np.max(affine_mean_reductions):.2f}%)",
            f"Random shared-vs-native-Taylor global reduction: "
            f"{100.0 * np.mean(native_global_reductions):.2f}% "
            f"(min {100.0 * np.min(native_global_reductions):.2f}%, "
            f"max {100.0 * np.max(native_global_reductions):.2f}%)",
            f"Random shared-vs-native-Taylor mean reduction: "
            f"{100.0 * np.mean(native_mean_reductions):.2f}% "
            f"(min {100.0 * np.min(native_mean_reductions):.2f}%, "
            f"max {100.0 * np.max(native_mean_reductions):.2f}%)",
            f"Random concretized-core vs Taylor mean reduction: "
            f"{100.0 * np.mean(concretized_native_mean_reductions):.2f}% "
            f"(min {100.0 * np.min(concretized_native_mean_reductions):.2f}%, "
            f"max {100.0 * np.max(concretized_native_mean_reductions):.2f}%)",
            f"Crossing-pair mean reduction vs Taylor: "
            f"{100.0 * np.mean(native_mean_by_category['crossing']):.2f}%",
            f"Fixed-sign-pair mean reduction vs Taylor: "
            f"{100.0 * np.mean(native_mean_by_category['fixed-sign']):.2f}%",
            f"Sign-crossing inputs: {crossing_cases}/40",
            f"Minimum sampled containment margin: {minimum_margin:.3e}",
            "The retained-core shared-Q rule and native Taylor multiplication "
            "are both O(n^2); no timing benchmark was run.",
            "Sampling is a regression/tightness diagnostic, not the soundness proof.",
            "No production behavior changed.",
        )
    )
    print("\n\n" + "\n".join(summary))


# The following comparison uses the same original input fibers for all four
# methods.  These helpers remain test-local; no arithmetic is registered.
def _pair_from_shared(bound: SharedQuadraticBound):
    return _endpoint_models(bound)


def _pair_range(pair):
    return Interval(
        polynomial_range(pair[0]).lower,
        polynomial_range(pair[1]).upper,
    )


def _assert_nonnegative_pair(pair):
    lower, upper = pair
    assert float(polynomial_range(upper - lower).lower) >= -_TOLERANCE
    assert float(polynomial_range(lower).lower) >= -_TOLERANCE
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for model in pair
        for leaf in jax.tree.leaves(model)
    )


def _pair_to_taylor(pair):
    lower, upper = pair
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    half_range = polynomial_range(half_width)
    radius = jnp.maximum(jnp.abs(half_range.lower), jnp.abs(half_range.upper))
    return _model(
        midpoint.constant, midpoint.linear, midpoint.quadratic, (-radius, radius)
    )


def _pair_to_shared(pair):
    """Enclose a quadratic half-width by an affine upper function.

    D = d0 + d1 @ xi + q(xi) <= d0 + d1 @ xi + max(q).  The
    quadratic-only range is analytic, not a sampled fit.  For an ordered
    pair D >= 0, so the enclosing affine half-width is also nonnegative.
    """

    lower, upper = pair
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    quadratic_only = _model(
        0.0, jnp.zeros_like(half_width.linear), half_width.quadratic
    )
    affine_upper_constant = (
        half_width.constant + polynomial_range(quadratic_only).upper
    )
    return _symmetric_bound(
        midpoint, affine_upper_constant, half_width.linear
    )


def _interval_overflow_pair(first, second):
    """Generic O(n^2) Taylor overflow on each needed endpoint product."""

    lower, _ = taylor_endpoint_models(first[0] * second[0])
    _, upper = taylor_endpoint_models(first[1] * second[1])
    return lower, upper


def _am_gm_pair(first, second):
    output = nonnegative_endpoint_product(
        QuadraticEndpointPair(*first, jnp.asarray(True)),
        QuadraticEndpointPair(*second, jnp.asarray(True)),
    )
    return output


def _fiber_metrics(representation, sources, reference, kind):
    if kind == "pair":
        fibers = _evaluate_pair(representation, sources)
        global_range = _pair_range(representation)
    elif kind == "shared":
        fibers = _evaluate_bound(representation, sources)
        global_range = _bound_range(representation)
    else:
        fibers = _evaluate_taylor(representation, sources)
        global_range = taylor_range(representation)
    widths = fibers[:, 1] - fibers[:, 0]
    exact_widths = reference[:, 1] - reference[:, 0]
    lower_margin = reference[:, 0] - fibers[:, 0]
    upper_margin = fibers[:, 1] - reference[:, 1]
    result = dict(
        lower=float(global_range.lower),
        upper=float(global_range.upper),
        global_width=float(global_range.width),
        mean_width=float(np.mean(widths)),
        max_width=float(np.max(widths)),
        mean_excess=float(np.mean(widths - exact_widths)),
        max_excess=float(np.max(widths - exact_widths)),
        min_lower_margin=float(np.min(lower_margin)),
        min_upper_margin=float(np.min(upper_margin)),
    )
    assert np.all(np.isfinite(list(result.values())))
    assert result["min_lower_margin"] >= -_TOLERANCE
    assert result["min_upper_margin"] >= -_TOLERANCE
    return result


def _evaluate_pair(pair, sources):
    return np.column_stack(
        tuple(
            np.asarray(jax.vmap(lambda z: evaluate_polynomial(model, z))(sources))
            for model in pair
        )
    )


def _original_product_fibers(first, second, sources):
    x = _evaluate_pair(first, sources)
    y = _evaluate_pair(second, sources)
    corners = np.stack(
        (x[:, 0] * y[:, 0], x[:, 0] * y[:, 1],
         x[:, 1] * y[:, 0], x[:, 1] * y[:, 1]), axis=1
    )
    return np.column_stack((corners.min(axis=1), corners.max(axis=1)))


def _positive_midpoint(rng, n, random):
    if random:
        linear = rng.normal(scale=0.13, size=n)
        raw = rng.normal(scale=0.07, size=(n, n))
        quadratic = 0.5 * (raw + raw.T)
    else:
        linear = np.resize([0.22, -0.17, 0.11], n)
        quadratic = np.diag(np.resize([0.12, -0.08, 0.06], n))
    return _model(0.0, linear, quadratic)


def _comparison_input(family, rng, n, random, second):
    midpoint = _positive_midpoint(rng, n, random)
    if second:
        midpoint = _model(
            midpoint.constant,
            -0.8 * midpoint.linear,
            -0.6 * midpoint.quadratic,
        )
    if family == "Taylor":
        low, high = (-0.07, 0.11) if not random else (
            -rng.uniform(0.03, 0.10), rng.uniform(0.04, 0.12)
        )
        provisional = _model(0.0, midpoint.linear, midpoint.quadratic, (low, high))
        shift = 0.4 - float(taylor_range(provisional).lower)
        original_taylor = _model(
            shift, midpoint.linear, midpoint.quadratic, (low, high)
        )
        original_pair = taylor_endpoint_models(original_taylor)
        shared = _symmetric_bound(
            _model(shift + 0.5 * (low + high), midpoint.linear, midpoint.quadratic),
            0.5 * (high - low),
            np.zeros(n),
        )
        return original_pair, shared, original_taylor
    if family == "Shared":
        affine = (rng.normal(scale=0.02, size=n) if random
                  else np.resize([0.025, -0.018, 0.012], n))
        width_constant = np.sum(np.abs(affine)) + 0.045
        shared = _symmetric_bound(midpoint, width_constant, affine)
        shift = 0.4 - float(_bound_range(shared).lower)
        shared = _shift_bound(shared, shift)
        original_pair = _pair_from_shared(shared)
        return original_pair, shared, _bound_to_taylor(shared)
    # Full-pair half-width has a genuinely nonzero quadratic matrix.
    affine = (rng.normal(scale=0.012, size=n) if random
              else np.resize([0.012, -0.008, 0.006], n))
    diagonal = (rng.uniform(0.025, 0.065, size=n) if random
                else np.resize([0.05, 0.035, 0.04], n))
    half_width = _model(
        np.sum(np.abs(affine)) + 0.035,
        affine,
        np.diag(2.0 * diagonal),
    )
    provisional_lower = midpoint - half_width
    shift = 0.4 - float(polynomial_range(provisional_lower).lower)
    shifted_midpoint = midpoint + shift
    original_pair = (shifted_midpoint - half_width,
                     shifted_midpoint + half_width)
    return original_pair, _pair_to_shared(original_pair), _pair_to_taylor(original_pair)


def _format_comparison_row(name, result, baseline):
    return (
        f"{name}: [{result['lower']:.5f},{result['upper']:.5f}] "
        f"W={result['global_width']:.5f} "
        f"fiber={result['mean_width']:.5f}/{result['max_width']:.5f} "
        f"excess={result['mean_excess']:.5f}/{result['max_excess']:.5f} "
        f"margins={result['min_lower_margin']:.2e}/"
        f"{result['min_upper_margin']:.2e} "
        f"ratio(T)={result['global_width']/baseline['global_width']:.3f}/"
        f"{result['mean_width']/baseline['mean_width']:.3f}"
    )


def test_aligned_nonnegative_three_representation_multiplication():
    """Compare four separate rules on identical original product fibers."""

    rng = np.random.default_rng(20260930)
    counts = {name: {key: {outcome: 0 for outcome in ('win', 'loss', 'tie')}
                     for key in ('global_width', 'mean_width')}
              for name in ('Shared', 'O(n^2) pair', 'AM-GM pair')}
    pair_counts = {key: {outcome: 0 for outcome in ('win', 'loss', 'tie')}
                   for key in ('global_width', 'mean_width')}
    lines = []
    for n in (2, 3):
        sources = np.concatenate(
            (np.array(list(itertools.product((-1.0, 1.0), repeat=n))),
             rng.uniform(-1.0, 1.0, size=(65, n))), axis=0
        )
        for family in ('Taylor', 'Shared', 'Full pair'):
            for random in (False, True):
                first = _comparison_input(family, rng, n, random, False)
                second = _comparison_input(family, rng, n, random, True)
                original = (first[0], second[0])
                for pair in original:
                    _assert_nonnegative_pair(pair)
                reference = _original_product_fibers(*original, sources)
                inputs = {
                    'original': original,
                    'shared': (first[1], second[1]),
                    'Taylor': (first[2], second[2]),
                }
                lines.append(f"n={n} {family} {'random' if random else 'control'}")
                for index in (0, 1):
                    baseline_input = _fiber_metrics(
                        inputs['original'][index], sources,
                        _evaluate_pair(inputs['original'][index], sources), 'pair'
                    )
                    details = []
                    for label, kind in (('original', 'pair'),
                                        ('shared', 'shared'), ('Taylor', 'Taylor')):
                        measure = _fiber_metrics(
                            inputs[label][index], sources,
                            _evaluate_pair(inputs['original'][index], sources), kind
                        )
                        details.append(
                            f"{label} W={measure['global_width']:.5f} "
                            f"fiber={measure['mean_width']:.5f} "
                            f"added={measure['global_width']-baseline_input['global_width']:.5f}/"
                            f"{measure['mean_width']-baseline_input['mean_width']:.5f}"
                        )
                    lines.append(f"  input {index+1}: " + '; '.join(details))
                am_gm = _am_gm_pair(*original)
                assert bool(am_gm.preconditions_hold)
                output = {
                    'Taylor': (inputs['Taylor'][0] * inputs['Taylor'][1], 'Taylor'),
                    'Shared': (_retained_core_product(*inputs['shared']), 'shared'),
                    'O(n^2) pair': (_interval_overflow_pair(*original), 'pair'),
                    'AM-GM pair': ((am_gm.lower, am_gm.upper), 'pair'),
                }
                measures = {
                    name: _fiber_metrics(value, sources, reference, kind)
                    for name, (value, kind) in output.items()
                }
                baseline = measures['Taylor']
                for name, result in measures.items():
                    lines.append('  ' + _format_comparison_row(name, result, baseline))
                    if name != 'Taylor':
                        for key in ('global_width', 'mean_width'):
                            counts[name][key][_outcome(result, baseline, key)] += 1
                for key in ('global_width', 'mean_width'):
                    pair_counts[key][_outcome(
                        measures['O(n^2) pair'], measures['AM-GM pair'], key
                    )] += 1
                for name in ('O(n^2) pair', 'AM-GM pair'):
                    converted = _fiber_metrics(
                        _pair_to_taylor(output[name][0]), sources, reference, 'Taylor'
                    )
                    raw = measures[name]
                    lines.append(
                        f"  {name} -> Taylor conversion added "
                        f"W={converted['global_width']-raw['global_width']:.5f} "
                        f"fiber={converted['mean_width']-raw['mean_width']:.5f}"
                    )
                lines.append(
                    f"  O(n^2)-minus-AM-GM raw W="
                    f"{measures['O(n^2) pair']['global_width']-measures['AM-GM pair']['global_width']:.5f} "
                    f"fiber={measures['O(n^2) pair']['mean_width']-measures['AM-GM pair']['mean_width']:.5f}"
                )
    for name, comparisons in counts.items():
        for key, outcomes in comparisons.items():
            lines.append(f"{name} vs Taylor {key} W/L/T: "
                         f"{outcomes['win']}/{outcomes['loss']}/{outcomes['tie']}")
    for key, outcomes in pair_counts.items():
        lines.append(f"O(n^2) vs AM-GM {key} W/L/T: "
                     f"{outcomes['win']}/{outcomes['loss']}/{outcomes['tie']}")
    lines.append('Sampling checks containment and tightness, not continuous soundness.')
    print('\n' + '\n'.join(lines))


def test_interval_overflow_pair_jit_shape_smoke():
    rng = np.random.default_rng(20260930)
    first = _comparison_input('Full pair', rng, 8, True, False)[0]
    second = _comparison_input('Full pair', rng, 8, True, True)[0]
    for pair in (first, second):
        _assert_nonnegative_pair(pair)
    eager = _interval_overflow_pair(first, second)
    compiled = jax.jit(_interval_overflow_pair)(first, second)
    for expected, observed in zip(eager, compiled):
        for expected_leaf, observed_leaf in zip(
            jax.tree.leaves(expected), jax.tree.leaves(observed)
        ):
            np.testing.assert_allclose(observed_leaf, expected_leaf, atol=1e-12)
    assert eager[0].quadratic.shape == (8, 8)
    assert eager[1].quadratic.shape == (8, 8)
