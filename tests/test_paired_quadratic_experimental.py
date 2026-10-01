"""Focused contract checks for opt-in quadratic endpoint propagation."""

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion import (
    PairedQuadratic,
    constant_taylor_model,
    evaluate_pair,
    normalized_pair_seed,
    pair_from_taylor,
    pair_range,
    pair_status,
    pair_to_taylor,
    pqif,
    taylor_range,
)
from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import TaylorModel

jax.config.update("jax_enable_x64", True)


def _variable_pair():
    center = TaylorModel(
        jnp.array([1.2, 1.7]),
        jnp.array([[0.2, -0.1], [-0.15, 0.08]]),
        jnp.array([[[0.12, 0.03], [0.03, -0.04]],
                   [[0.06, -0.02], [-0.02, 0.05]]]),
        Interval(jnp.zeros(2), jnp.zeros(2)),
    )
    width = TaylorModel(
        jnp.array([0.15, 0.12]),
        jnp.zeros((2, 2)),
        jnp.array([[[0.08, 0.0], [0.0, 0.04]],
                   [[0.04, 0.0], [0.0, 0.06]]]),
        Interval(jnp.zeros(2), jnp.zeros(2)),
    )
    return PairedQuadratic(center - width, center + width)


def _assert_pair_close(first, second):
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(a, b, atol=2e-12)


def test_seed_range_order_and_exact_linear_operations():
    seed = normalized_pair_seed(jnp.array([-0.02, -0.03]), jnp.array([0.02, 0.03]))
    assert bool(jnp.all(pair_status(seed)["ordered"]))
    np.testing.assert_allclose(pair_range(seed).lower, [-0.02, -0.03])
    np.testing.assert_allclose(pair_range(seed).upper, [0.02, 0.03])
    transformed = pqif(lambda x: jnp.reshape(2.0 * x - jnp.array([0.01, 0.02]), (2,)))(seed)
    assert bool(jnp.all(pair_status(transformed)["ordered"]))
    for source in (jnp.array([-1.0, 1.0]), jnp.array([0.3, -0.4])):
        physical = jnp.array([0.02, 0.03]) * source
        lower, upper = evaluate_pair(transformed, source)
        expected = 2.0 * physical - jnp.array([0.01, 0.02])
        np.testing.assert_allclose(lower, expected, atol=1e-15)
        np.testing.assert_allclose(upper, expected, atol=1e-15)


def test_nonnegative_product_conversion_and_fallback_eager_jit_vmap():
    first = _variable_pair()
    second = first + 0.3
    assert bool(jnp.all(pair_status(first)["ordered"]))
    product = first * second
    assert bool(jnp.all(pair_status(product)["ordered"]))
    compiled = jax.jit(lambda x, y: x * y)(first, second)
    _assert_pair_close(product, compiled)

    sources = jnp.array([[-1.0, -1.0], [-0.4, 0.7], [0.0, 0.0], [0.8, -0.2], [1.0, 1.0]])
    lower, upper = jax.vmap(lambda z: evaluate_pair(product, z))(sources)
    x_lower, x_upper = jax.vmap(lambda z: evaluate_pair(first, z))(sources)
    y_lower, y_upper = jax.vmap(lambda z: evaluate_pair(second, z))(sources)
    assert bool(jnp.all(lower <= x_lower * y_lower + 2e-12))
    assert bool(jnp.all(upper >= x_upper * y_upper - 2e-12))

    converted = pair_to_taylor(first)
    center_lower, center_upper = jax.vmap(lambda z: evaluate_pair(first, z))(sources)
    from immrax.inclusion import evaluate_polynomial
    polynomial = jax.vmap(lambda z: evaluate_polynomial(converted, z))(sources)
    assert bool(jnp.all(polynomial + converted.remainder.lower <= center_lower + 2e-12))
    assert bool(jnp.all(polynomial + converted.remainder.upper >= center_upper - 2e-12))

    transform = pqif(lambda x: jnp.sin(x))
    eager = transform(first)
    _assert_pair_close(eager, jax.jit(transform)(first))
    vmapped = jax.vmap(transform)(first)
    assert vmapped.shape == first.shape
    for result in (eager, vmapped):
        assert bool(jnp.all(pair_status(result)["ordered"]))
        low, high = jax.vmap(lambda z: evaluate_pair(result, z))(sources)
        values = jnp.sin(0.5 * (center_lower + center_upper))
        assert bool(jnp.all(low <= values + 2e-12))
        assert bool(jnp.all(high >= values - 2e-12))


def test_invalid_order_is_visible_to_conversion():
    good = constant_taylor_model(jnp.array(1.0), 2)
    bad = PairedQuadratic(good, good - 0.25)
    assert not bool(jnp.all(pair_status(bad)["ordered"]))
    assert bool(jnp.isnan(pair_to_taylor(bad).constant))
