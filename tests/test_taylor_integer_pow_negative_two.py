"""Focused checks for the experimental Taylor ``integer_pow(y=-2)`` rule."""

import itertools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    TaylorModel,
    _integer_pow,
    _square,
    _unary_second_order,
    constant_taylor_model,
    evaluate_polynomial,
    normalized_taylor_seed,
    taylor_range,
    tmif,
)

jax.config.update("jax_enable_x64", True)


def _model(constant, linear, quadratic, remainder):
    return TaylorModel(
        jnp.asarray(constant, dtype=float),
        jnp.asarray(linear, dtype=float),
        jnp.asarray(quadratic, dtype=float),
        Interval(jnp.asarray(remainder[0], dtype=float), jnp.asarray(remainder[1], dtype=float)),
    )


def _leaves(model):
    return (model.constant, model.linear, model.quadratic, model.remainder.lower, model.remainder.upper)


def _assert_no_nan(model):
    for leaf in _leaves(model):
        assert not np.any(np.isnan(np.asarray(leaf)))


def _assert_top(model):
    _assert_no_nan(model)
    bounds = taylor_range(model)
    assert np.all(np.asarray(bounds.lower) == -np.inf)
    assert np.all(np.asarray(bounds.upper) == np.inf)


def _assert_same(actual, expected):
    for a, b in zip(_leaves(actual), _leaves(expected)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# Two sources with nonzero linear, quadratic (including off-diagonal), and
# remainder terms; every represented value lies in [0.58, 1.47].
POSITIVE = _model(
    [1.0, 0.8],
    [[0.2, -0.1], [0.05, 0.12]],
    [[[0.1, -0.04], [-0.04, 0.06]], [[-0.03, 0.02], [0.02, 0.05]]],
    ([-0.03, -0.01], [0.04, 0.02]),
)


def _source_grid(count=17):
    axis = np.linspace(-1.0, 1.0, count)
    return np.asarray(list(itertools.product(axis, axis)))


def _assert_pointwise_contains(result, argument, function):
    """``f(P(xi) + r) - P_f(xi)`` lies in the result remainder on a grid."""
    sources = jnp.asarray(_source_grid())
    argument_poly = jax.vmap(lambda s: evaluate_polynomial(argument, s))(sources)
    result_poly = jax.vmap(lambda s: evaluate_polynomial(result, s))(sources)
    lower, upper = np.asarray(argument.remainder.lower), np.asarray(argument.remainder.upper)
    for t in np.linspace(0.0, 1.0, 11):
        r = lower + t * (upper - lower)
        errors = np.asarray(function(argument_poly + r) - result_poly)
        assert np.all(errors >= np.asarray(result.remainder.lower) - 1e-12)
        assert np.all(errors <= np.asarray(result.remainder.upper) + 1e-12)


def test_point_inputs_and_exact_constants():
    np.testing.assert_allclose(_integer_pow(jnp.asarray(2.0), y=-2), 0.25, rtol=0)
    np.testing.assert_allclose(_integer_pow(jnp.array([0.5, 4.0]), y=-2), [4.0, 0.0625], rtol=0)
    constant = constant_taylor_model(jnp.array([2.0, 0.5]), 3)
    result = _integer_pow(constant, y=-2)
    np.testing.assert_allclose(result.constant, [0.25, 4.0], rtol=1e-15)
    np.testing.assert_array_equal(result.linear, 0.0)
    np.testing.assert_array_equal(result.quadratic, 0.0)
    bounds = taylor_range(result)
    np.testing.assert_allclose(bounds.lower, [0.25, 4.0], rtol=1e-15)
    np.testing.assert_allclose(bounds.upper, [0.25, 4.0], rtol=1e-15)


def test_jaxpr_reaches_integer_pow_minus_two():
    eqns = jax.make_jaxpr(lambda u: u**-2)(1.0).jaxpr.eqns
    assert [(e.primitive, e.params.get("y")) for e in eqns] == [(lax.integer_pow_p, -2)]


def test_positive_taylor_range_contains_source_grid():
    result = tmif(lambda u: u**-2)(POSITIVE)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in _leaves(result))
    _assert_pointwise_contains(result, POSITIVE, lambda u: u**-2.0)
    # Expansion values at the center.
    np.testing.assert_allclose(result.constant, np.array([1.0, 0.8]) ** -2, rtol=1e-15)
    np.testing.assert_allclose(
        result.linear, (-2 * np.array([1.0, 0.8]) ** -3)[:, None] * np.asarray(POSITIVE.linear), rtol=1e-15
    )
    # Range encloses the true range implied by the argument's range.
    arg = taylor_range(POSITIVE)
    bounds = taylor_range(result)
    assert np.all(np.asarray(bounds.lower) <= np.asarray(arg.upper) ** -2.0 + 1e-15)
    assert np.all(np.asarray(bounds.upper) >= np.asarray(arg.lower) ** -2.0 - 1e-15)


def test_composed_source_dependence_contains_samples():
    seed = normalized_taylor_seed(jnp.array([-0.2, -0.15]), jnp.array([0.2, 0.25]))
    function = lambda z: (1.3 + 0.2 * z[0] + z[1] ** 2 - 0.1 * z[0] * z[1]) ** -2
    result = tmif(function)(seed)
    sources = jnp.asarray(_source_grid())
    physical = jnp.asarray(seed.constant) + sources * (0.5 * jnp.array([0.4, 0.4]))
    samples = np.asarray(jax.vmap(function)(physical))
    bounds = taylor_range(result)
    assert np.all(samples >= np.asarray(bounds.lower) - 1e-12)
    assert np.all(samples <= np.asarray(bounds.upper) + 1e-12)


def test_regression_against_two_reciprocal_applications():
    direct = _integer_pow(POSITIVE, y=-2)
    reciprocal = _unary_second_order(POSITIVE, "reciprocal")
    twice = _square(reciprocal)
    np.testing.assert_allclose(direct.constant, twice.constant, rtol=1e-14)
    np.testing.assert_allclose(direct.linear, twice.linear, rtol=1e-14)
    _assert_pointwise_contains(twice, POSITIVE, lambda u: u**-2.0)
    a, b = taylor_range(direct), taylor_range(twice)
    width_ratio = np.asarray(a.width) / np.asarray(b.width)
    assert np.all((width_ratio > 0.5) & (width_ratio < 2.0))


@pytest.mark.parametrize(
    "argument",
    [
        _model([0.5], [[0.5]], [[[0.0]]], ([0.0], [0.0])),  # touches zero: [0, 1]
        _model([0.0], [[1.0]], [[[0.0]]], ([0.0], [0.0])),  # crosses zero: [-1, 1]
        _model([0.5], [[0.2]], [[[0.0]]], ([-0.3], [0.0])),  # remainder reaches zero
        _model([-2.0], [[0.5]], [[[0.0]]], ([0.0], [0.0])),  # negative: conservative top
    ],
)
def test_nonpositive_ranges_return_top_without_nan(argument):
    _assert_top(_integer_pow(argument, y=-2))
    _assert_top(jax.jit(lambda m: _integer_pow(m, y=-2))(argument))


def test_eager_and_jit_agree():
    transform = tmif(lambda u: u**-2)
    eager, jitted = transform(POSITIVE), jax.jit(transform)(POSITIVE)
    # XLA fusion may change the last bit; the rule itself is identical.
    for a, b in zip(_leaves(jitted), _leaves(eager)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14, atol=1e-16)


def test_existing_exponents_unchanged():
    _assert_same(_integer_pow(POSITIVE, y=0), constant_taylor_model(jnp.ones_like(POSITIVE.constant), POSITIVE))
    _assert_same(_integer_pow(POSITIVE, y=1), POSITIVE)
    _assert_same(_integer_pow(POSITIVE, y=2), _square(POSITIVE))
    for exponent in (-1, -3, 3):
        with pytest.raises(NotImplementedError):
            _integer_pow(POSITIVE, y=exponent)
