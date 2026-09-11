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
        jnp.array([1.0]), 0.0, jnp.array([1.0]), 0.0
    )
    negative_square = template.product_of_source_planes(
        jnp.array([1.0]), 0.0, jnp.array([-1.0]), 0.0
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

    affine = template.product_of_source_planes(first_coeff, 0.0, second_coeff, 0.0)
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

    assert product.shape == (2,)
    assert broadcast_product.shape == (2,)
    assert jnp.all(affine_lower <= exact + 2e-6)
    assert jnp.all(affine_upper >= exact - 2e-6)
    assert jnp.all(interval_lower <= exact + 2e-6)
    assert jnp.all(interval_upper >= exact - 2e-6)
    assert jnp.all(best_lower <= exact + 2e-6)
    assert jnp.all(best_upper >= exact - 2e-6)
    assert jnp.all(best_product.width <= product.width + 2e-6)
    assert jnp.all(best_product.width <= interval_product.width + 2e-6)


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
