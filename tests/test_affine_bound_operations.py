import jax
import jax.numpy as jnp

import immrax as irx


def _bound(lower_coeff, lower_bias, upper_coeff, upper_bias, lower, upper):
    return irx.AffineBound(
        jnp.asarray(lower_coeff),
        jnp.asarray(lower_bias),
        jnp.asarray(upper_coeff),
        jnp.asarray(upper_bias),
        jnp.asarray(lower),
        jnp.asarray(upper),
    )


def test_plane_extrema_and_eliminate_prefix():
    value = _bound(
        [2.0, -1.0],
        0.5,
        [-3.0, 0.5],
        1.0,
        [-1.0, -2.0],
        [2.0, 3.0],
    )

    lower, upper = value.plane_extrema(jnp.array([2.0, -1.0]), jnp.array(0.5))
    reduced = value.eliminate_prefix(1)

    assert jnp.allclose(lower, -4.5)
    assert jnp.allclose(upper, 6.5)
    assert reduced.input_size == 1
    assert jnp.allclose(reduced.lower_coeff, jnp.array([-1.0]))
    assert jnp.allclose(reduced.lower_bias, -1.5)
    assert jnp.allclose(reduced.upper_coeff, jnp.array([0.5]))
    assert jnp.allclose(reduced.upper_bias, 4.0)
    assert jnp.allclose(reduced.lower, value.lower)
    assert jnp.allclose(reduced.upper, value.upper)


def test_product_of_source_planes_uses_square_envelopes():
    template = irx.constant_affine_bound(0.0, jnp.array([-1.0]), jnp.array([2.0]))

    square = template.product_of_source_planes(
        jnp.array([1.0]), 0.0, jnp.array([1.0]), 0.0,
        quadratic_relaxation="affine",
    )
    negative_square = template.product_of_source_planes(
        jnp.array([1.0]), 0.0, jnp.array([-1.0]), 0.0,
        quadratic_relaxation="affine",
    )
    interval_square = template.product_of_source_planes(
        jnp.array([1.0]),
        0.0,
        jnp.array([1.0]),
        0.0,
        quadratic_relaxation="interval",
    )

    assert jnp.allclose(square.lower_coeff, jnp.array([0.0]))
    assert jnp.allclose(square.lower_bias, 0.0)
    assert jnp.allclose(square.upper_coeff, jnp.array([1.0]))
    assert jnp.allclose(square.upper_bias, 2.0)
    assert jnp.allclose(negative_square.lower_coeff, jnp.array([-1.0]))
    assert jnp.allclose(negative_square.lower_bias, -2.0)
    assert jnp.allclose(negative_square.upper_coeff, jnp.array([0.0]))
    assert jnp.allclose(negative_square.upper_bias, 0.0)
    assert jnp.allclose(interval_square.lower_coeff, jnp.array([0.0]))
    assert jnp.allclose(interval_square.lower_bias, 0.0)
    assert jnp.allclose(interval_square.upper_coeff, jnp.array([0.0]))
    assert jnp.allclose(interval_square.upper_bias, 4.0)


def test_interval_quadratic_relaxation_can_tighten_cross_term_concretization():
    template = irx.constant_affine_bound(0.0, -jnp.ones(2), jnp.ones(2))
    first_coeff = jnp.array([1.0, 0.0])
    second_coeff = jnp.array([0.0, 1.0])

    affine = template.product_of_source_planes(
        first_coeff, 0.0, second_coeff, 0.0,
        quadratic_relaxation="affine",
    )
    interval = template.product_of_source_planes(
        first_coeff,
        0.0,
        second_coeff,
        0.0,
        quadratic_relaxation="interval",
    )
    best = template.product_of_source_planes(
        first_coeff,
        0.0,
        second_coeff,
        0.0,
        quadratic_relaxation="best",
    )

    assert jnp.allclose(affine.lower, -3.0)
    assert jnp.allclose(affine.upper, 3.0)
    assert jnp.allclose(interval.lower, -1.0)
    assert jnp.allclose(interval.upper, 1.0)
    assert jnp.allclose(best.lower, interval.lower)
    assert jnp.allclose(best.upper, interval.upper)


def test_zonotope_source_product_joint_remainder_and_interval_fallback():
    centered = irx.constant_affine_bound(0.0, -jnp.ones(2), jnp.ones(2))
    coeff = jnp.ones(2)
    interval = centered.product_of_source_planes(
        coeff, 0.0, coeff, 0.0, quadratic_relaxation="interval"
    )
    zonotope = centered.product_of_source_planes(
        coeff, 0.0, coeff, 0.0, quadratic_relaxation="zonotope"
    )
    assert jnp.allclose(interval.lower, -2.0)
    assert jnp.allclose(zonotope.lower, 0.0)
    assert jnp.allclose(zonotope.upper, 4.0)

    shifted = irx.constant_affine_bound(0.0, jnp.array([1.0]), jnp.array([2.0]))
    shifted_interval = shifted.product_of_source_planes(
        jnp.array([1.0]), 0.0, jnp.array([1.0]), 0.0,
        quadratic_relaxation="interval",
    )
    shifted_zonotope = shifted.product_of_source_planes(
        jnp.array([1.0]), 0.0, jnp.array([1.0]), 0.0,
        quadratic_relaxation="zonotope",
    )
    assert jnp.allclose(shifted_zonotope.lower, shifted_interval.lower)
    assert jnp.allclose(shifted_zonotope.upper, shifted_interval.upper)


def test_product_of_source_planes_is_jittable_and_sound_for_batched_planes():
    domain_lower = jnp.array([-1.0, -2.0, 0.5])
    domain_upper = jnp.array([2.0, 1.0, 1.5])
    template = irx.constant_affine_bound(0.0, domain_lower, domain_upper)
    first_coeff = jnp.array([[1.0, -0.5, 0.25], [-2.0, 0.3, 1.2]])
    first_bias = jnp.array([0.4, -0.7])
    second_coeff = jnp.array([[-0.2, 1.5, 0.8], [0.6, -1.0, 0.1]])
    second_bias = jnp.array([1.1, 0.2])

    product = jax.jit(template.product_of_source_planes)(
        first_coeff, first_bias, second_coeff, second_bias
    )
    interval_product = jax.jit(
        lambda first_coeff, first_bias, second_coeff, second_bias: (
            template.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="interval",
            )
        )
    )(first_coeff, first_bias, second_coeff, second_bias)
    zonotope_product = jax.jit(
        lambda first_coeff, first_bias, second_coeff, second_bias: (
            template.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="zonotope",
            )
        )
    )(first_coeff, first_bias, second_coeff, second_bias)
    best_product = jax.jit(
        lambda first_coeff, first_bias, second_coeff, second_bias: (
            template.product_of_source_planes(
                first_coeff,
                first_bias,
                second_coeff,
                second_bias,
                quadratic_relaxation="best",
            )
        )
    )(first_coeff, first_bias, second_coeff, second_bias)
    broadcast_product = template.product_of_source_planes(
        first_coeff, 0.4, second_coeff[0], 1.1
    )
    axes = [
        jnp.linspace(lower, upper, 9)
        for lower, upper in zip(domain_lower, domain_upper, strict=True)
    ]
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    exact = (points @ first_coeff.T + first_bias) * (
        points @ second_coeff.T + second_bias
    )
    affine_lower = points @ product.lower_coeff.T + product.lower_bias
    affine_upper = points @ product.upper_coeff.T + product.upper_bias
    interval_lower = (
        points @ interval_product.lower_coeff.T + interval_product.lower_bias
    )
    interval_upper = (
        points @ interval_product.upper_coeff.T + interval_product.upper_bias
    )
    best_lower = points @ best_product.lower_coeff.T + best_product.lower_bias
    best_upper = points @ best_product.upper_coeff.T + best_product.upper_bias
    zonotope_lower = (
        points @ zonotope_product.lower_coeff.T + zonotope_product.lower_bias
    )
    zonotope_upper = (
        points @ zonotope_product.upper_coeff.T + zonotope_product.upper_bias
    )

    assert product.shape == (2,)
    assert broadcast_product.shape == (2,)
    assert jnp.all(affine_lower <= exact + 2e-6)
    assert jnp.all(affine_upper >= exact - 2e-6)
    assert jnp.all(interval_lower <= exact + 2e-6)
    assert jnp.all(interval_upper >= exact - 2e-6)
    assert jnp.all(zonotope_lower <= exact + 2e-6)
    assert jnp.all(zonotope_upper >= exact - 2e-6)
    assert jnp.all(best_lower <= exact + 2e-6)
    assert jnp.all(best_upper >= exact - 2e-6)
    assert jnp.all(best_product.width <= product.width + 2e-6)
    assert jnp.all(best_product.width <= interval_product.width + 2e-6)
    assert jnp.all(zonotope_product.lower >= interval_product.lower - 2e-6)
    assert jnp.all(zonotope_product.upper <= interval_product.upper + 2e-6)
    assert jnp.all(best_product.lower >= zonotope_product.lower - 2e-6)
    assert jnp.all(best_product.upper <= zonotope_product.upper + 2e-6)


def test_source_plane_times_square_interval_cubic_is_jittable_and_sound():
    domain_lower = jnp.array([-0.7, -1.2, 0.1])
    domain_upper = jnp.array([1.3, 0.8, 1.1])
    template = irx.constant_affine_bound(0.0, domain_lower, domain_upper)
    first_coeff = jnp.array([[1.0, -0.5, 0.25], [-0.2, 0.3, 1.2]])
    first_bias = jnp.array([0.4, 0.8])
    squared_coeff = jnp.array([[-0.2, 1.5, 0.8], [0.6, -1.0, 0.1]])
    squared_bias = jnp.array([1.1, -0.2])

    cubic = jax.jit(
        lambda first_coeff, first_bias, squared_coeff, squared_bias: (
            template.product_of_source_plane_and_square(
                first_coeff,
                first_bias,
                squared_coeff,
                squared_bias,
                cubic_relaxation="monomial_interval",
            )
        )
    )(first_coeff, first_bias, squared_coeff, squared_bias)
    grouped_cubic = template.product_of_source_plane_and_square(
        first_coeff,
        first_bias,
        squared_coeff,
        squared_bias,
        cubic_relaxation="grouped_interval",
    )
    zonotope_cubic = jax.jit(
        lambda first_coeff, first_bias, squared_coeff, squared_bias: (
            template.product_of_source_plane_and_square(
                first_coeff,
                first_bias,
                squared_coeff,
                squared_bias,
                cubic_relaxation="zonotope_remainder",
            )
        )
    )(first_coeff, first_bias, squared_coeff, squared_bias)
    broadcast_cubic = template.product_of_source_plane_and_square(
        first_coeff, 0.4, squared_coeff[0], 1.1
    )

    axes = [
        jnp.linspace(lower, upper, 9)
        for lower, upper in zip(domain_lower, domain_upper, strict=True)
    ]
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    exact = (points @ first_coeff.T + first_bias) * (
        points @ squared_coeff.T + squared_bias
    ) ** 2
    affine_lower = points @ cubic.lower_coeff.T + cubic.lower_bias
    affine_upper = points @ cubic.upper_coeff.T + cubic.upper_bias
    grouped_lower = points @ grouped_cubic.lower_coeff.T + grouped_cubic.lower_bias
    grouped_upper = points @ grouped_cubic.upper_coeff.T + grouped_cubic.upper_bias
    zonotope_lower = (
        points @ zonotope_cubic.lower_coeff.T + zonotope_cubic.lower_bias
    )
    zonotope_upper = (
        points @ zonotope_cubic.upper_coeff.T + zonotope_cubic.upper_bias
    )

    assert cubic.shape == (2,)
    assert broadcast_cubic.shape == (2,)
    assert jnp.all(affine_lower <= exact + 2e-6)
    assert jnp.all(affine_upper >= exact - 2e-6)
    assert jnp.all(grouped_lower <= exact + 2e-6)
    assert jnp.all(grouped_upper >= exact - 2e-6)
    assert jnp.all(zonotope_lower <= exact + 2e-6)
    assert jnp.all(zonotope_upper >= exact - 2e-6)


def test_source_plane_times_square_keeps_exact_first_order_taylor_term():
    domain_lower = jnp.array([-0.25, -0.5])
    domain_upper = jnp.array([0.75, 0.5])
    template = irx.constant_affine_bound(0.0, domain_lower, domain_upper)
    first_coeff = jnp.array([2.0, -1.0])
    first_bias = 0.3
    squared_coeff = jnp.array([-0.5, 1.5])
    squared_bias = 1.2

    cubic = template.product_of_source_plane_and_square(
        first_coeff, first_bias, squared_coeff, squared_bias
    )
    source_center = 0.5 * (domain_lower + domain_upper)
    first_center = source_center @ first_coeff + first_bias
    squared_center = source_center @ squared_coeff + squared_bias
    expected_linear = (
        squared_center**2 * first_coeff
        + 2.0 * first_center * squared_center * squared_coeff
    )

    assert jnp.allclose(cubic.lower_coeff, expected_linear)
    assert jnp.allclose(cubic.upper_coeff, expected_linear)


def test_monomial_cubic_retains_hinge_source_coefficient_cancellation():
    domain_lower = jnp.array([-0.02, -0.03])
    domain_upper = jnp.array([0.02, 0.03])
    template = irx.constant_affine_bound(0.0, domain_lower, domain_upper)
    penetration_coeff = jnp.array([-0.464954130, -1.36816796e-05])
    penetration_bias = 0.02209282
    shifted_bounce_coeff = jnp.array([0.01518907, -0.00136817])
    shifted_bounce_bias = 0.00153679

    grouped = template.product_of_source_plane_and_square(
        penetration_coeff,
        penetration_bias,
        shifted_bounce_coeff,
        shifted_bounce_bias,
        cubic_relaxation="grouped_interval",
    )
    monomial = template.product_of_source_plane_and_square(
        penetration_coeff,
        penetration_bias,
        shifted_bounce_coeff,
        shifted_bounce_bias,
        cubic_relaxation="monomial_interval",
    )
    zonotope = template.product_of_source_plane_and_square(
        penetration_coeff,
        penetration_bias,
        shifted_bounce_coeff,
        shifted_bounce_bias,
        cubic_relaxation="zonotope_remainder",
    )

    assert monomial.lower > grouped.lower
    assert monomial.upper < grouped.upper
    assert zonotope.lower >= monomial.lower
    assert zonotope.upper <= monomial.upper
    assert zonotope.width < monomial.width


def test_source_plane_products_support_an_empty_source_domain():
    empty = jnp.empty((0,))
    template = irx.constant_affine_bound(0.0, empty, empty)

    quadratic = template.product_of_source_planes(
        empty, 2.0, empty, -3.0, quadratic_relaxation="zonotope"
    )
    cubic = template.product_of_source_plane_and_square(
        empty, 2.0, empty, -3.0, cubic_relaxation="zonotope_remainder"
    )

    assert jnp.allclose(quadratic.lower, -6.0)
    assert jnp.allclose(quadratic.upper, -6.0)
    assert jnp.allclose(cubic.lower, 18.0)
    assert jnp.allclose(cubic.upper, 18.0)


def test_hull_inflate_and_containment_margin():
    first = _bound([1.0], 0.0, [1.0], 0.0, [-1.0], [1.0])
    second = _bound([-1.0], 0.0, [-1.0], 0.0, [-1.0], [1.0])
    hull = first.hull(second)
    inflated = hull.inflate(relative=0.5, absolute=0.25)

    assert jnp.allclose(hull.lower_coeff, jnp.array([0.0]))
    assert jnp.allclose(hull.lower_bias, -1.0)
    assert jnp.allclose(hull.upper_coeff, jnp.array([0.0]))
    assert jnp.allclose(hull.upper_bias, 1.0)
    assert jnp.allclose(inflated.lower, -1.75)
    assert jnp.allclose(inflated.upper, 1.75)
    assert jnp.allclose(inflated.inclusion_margin(hull), 0.75)
    assert bool(inflated.strictly_contains(hull, margin=0.74))
    assert not bool(inflated.strictly_contains(hull, margin=0.75))


def test_select_supports_jitted_scalar_condition():
    first = _bound([1.0], -1.0, [2.0], 3.0, [-1.0], [1.0])
    second = _bound([-1.0], 4.0, [-2.0], 5.0, [-1.0], [1.0])
    select = jax.jit(lambda condition: irx.AffineBound.select(condition, first, second))

    selected_first = select(jnp.asarray(True))
    selected_second = select(jnp.asarray(False))

    assert jnp.allclose(selected_first.lower_coeff, first.lower_coeff)
    assert jnp.allclose(selected_first.upper_bias, first.upper_bias)
    assert jnp.allclose(selected_second.lower_coeff, second.lower_coeff)
    assert jnp.allclose(selected_second.upper_bias, second.upper_bias)
