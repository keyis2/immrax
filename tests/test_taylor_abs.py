"""Focused checks for the two-sided range-aware Taylor rule for ``abs``.

Analytic assertions test the secant theorem; sampled containment is regression
evidence only.
"""

import itertools
from pathlib import Path

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion.interval import Interval
from immrax.inclusion.taylor import (
    TaylorModel,
    _abs,
    _max_or_min,
    _neg,
    _reduce_sum,
    evaluate_polynomial,
    taylor_inclusion_registry,
    taylor_range,
    tmif,
)

jax.config.update("jax_enable_x64", True)


def _model(constant, linear, quadratic=None, remainder=(0.0, 0.0), dtype=jnp.float64):
    constant = np.asarray(constant, dtype=float)
    linear = np.asarray(linear, dtype=float)
    n = linear.shape[-1]
    quadratic = np.zeros(linear.shape + (n,)) if quadratic is None else np.asarray(quadratic, dtype=float)
    lower = np.broadcast_to(np.asarray(remainder[0], dtype=float), constant.shape)
    upper = np.broadcast_to(np.asarray(remainder[1], dtype=float), constant.shape)
    return TaylorModel(jnp.asarray(constant, dtype), jnp.asarray(linear, dtype), jnp.asarray(quadratic, dtype),
                       Interval(jnp.asarray(lower, dtype), jnp.asarray(upper, dtype)))


def _leaves(model):
    return [np.asarray(v) for v in (model.constant, model.linear, model.quadratic,
                                    model.remainder.lower, model.remainder.upper)]


def _assert_same(actual, expected):
    for a, b in zip(_leaves(actual), _leaves(expected)):
        np.testing.assert_array_equal(a, b)


def _old_abs(x):
    """The previous rule, ``max(x, -x)`` through the ReLU crossing rule."""
    return _max_or_min(x, _neg(x), is_max=True)


def _assert_contains(result, argument, grid=9, steps=7, tol=1e-12):
    """``|P_x(xi) + r| - P_out(xi)`` lies in the output remainder for every entry."""
    n = argument.source_size
    if n:
        sources = np.array(list(itertools.product(np.linspace(-1, 1, grid), repeat=n)))
    else:
        sources = np.zeros((1, 0))
    xp = np.asarray(jax.vmap(lambda s: evaluate_polynomial(argument, s))(jnp.asarray(sources)))
    op = np.asarray(jax.vmap(lambda s: evaluate_polynomial(result, s))(jnp.asarray(sources)))
    lo, hi = np.asarray(argument.remainder.lower), np.asarray(argument.remainder.upper)
    olo, ohi = np.asarray(result.remainder.lower), np.asarray(result.remainder.upper)
    for t in np.linspace(0.0, 1.0, steps):
        err = np.abs(xp + lo + t * (hi - lo)) - op
        assert np.all(err >= olo - tol) and np.all(err <= ohi + tol)


# Crossing models with nonzero linear, quadratic (off-diagonal too), and
# asymmetric remainder terms; ``m`` is positive for the first, negative for the second.
POSITIVE_SLOPE = _model(0.3, [0.5, -0.2, 0.1], [[0.2, 0.05, 0], [0.05, -0.1, 0.02], [0, 0.02, 0.06]], (-0.04, 0.07))
NEGATIVE_SLOPE = _model(-0.25, [0.4, 0.1, -0.15], [[-0.1, 0.03, 0], [0.03, 0.08, 0], [0, 0, 0.05]], (-0.09, 0.02))


def test_fixed_sign_branches_are_exact_including_zero_endpoints():
    positive = _model(1.5, [0.2, 0.1], remainder=(-0.05, 0.08))
    negative = _model(-1.5, [0.2, 0.1], remainder=(-0.05, 0.08))
    touches_from_above = _model(1.0, [1.0])          # range [0, 2]
    touches_from_below = _model(-1.0, [1.0])         # range [-2, 0]
    zero = _model(0.0, [0.0, 0.0])
    for x in (positive, touches_from_above, zero):
        _assert_same(_abs(x), x)
    for x in (negative, touches_from_below):
        _assert_same(_abs(x), _neg(x))


def test_analytic_secant_coefficients_and_two_sided_remainder():
    x = _model(0.5, [1.0])  # range [-0.5, 1.5]: m = 1/2, b = 3/4
    out = _abs(x)
    assert float(out.constant) == 1.0 and float(out.linear[0]) == 0.5
    assert float(out.remainder.lower) == -0.75 and float(out.remainder.upper) == 0.0
    # Lower envelope m v is attained at v = 0, the upper m v + b at both endpoints.
    np.testing.assert_allclose([0.5 * v + 0.75 for v in (-0.5, 1.5)], [0.5, 1.5])


def test_negative_slope_scales_the_input_remainder_with_its_sign():
    x = _model(-0.1, [0.5], remainder=(-0.2, 0.1))  # range [-0.8, 0.5]
    l, u = -0.8, 0.5
    m, b = (u + l) / (u - l), -2 * l * u / (u - l)
    out = _abs(x)
    assert m < 0
    np.testing.assert_allclose(float(out.constant), m * -0.1 + b, rtol=1e-15)
    np.testing.assert_allclose(float(out.linear[0]), m * 0.5, rtol=1e-15)
    np.testing.assert_allclose(float(out.remainder.lower), m * 0.1 - b, rtol=1e-14)
    np.testing.assert_allclose(float(out.remainder.upper), m * -0.2, rtol=1e-14)
    _assert_contains(out, x)


@pytest.mark.parametrize("x", [POSITIVE_SLOPE, NEGATIVE_SLOPE, _model(0.0, [0.01], remainder=(-0.75, 0.75)),
                               _model(0.0, np.zeros(0), remainder=(-1.0, 2.0)),
                               _model(0.2, [1.0], [[0.3]], (-0.1, 0.05))],
                         ids=["positive_slope", "negative_slope", "remainder_dominated", "zero_sources", "one_source"])
def test_crossing_models_are_two_sided_enclosures(x):
    bounds = taylor_range(x)
    assert float(bounds.lower) < 0 < float(bounds.upper)
    out = _abs(x)
    assert out.source_size == x.source_size and out.shape == x.shape
    _assert_contains(out, x)
    # The rule is a genuine two-sided model: its lower bound is finite and below the upper.
    assert np.isfinite(float(out.remainder.lower)) and float(out.remainder.lower) < float(out.remainder.upper)


def test_mixed_sign_array_selects_entrywise():
    constant = np.array([[1.5, -1.5, 0.1], [0.0, -0.3, 2.0]])
    linear = np.zeros((2, 3, 2))
    linear[..., 0], linear[..., 1] = 0.2, -0.1
    linear[0, 2] = (0.7, 0.2)
    linear[1, 1] = (-0.4, 0.3)
    quadratic = np.zeros((2, 3, 2, 2))
    quadratic[0, 2] = [[0.2, 0.05], [0.05, 0.0]]
    x = _model(constant, linear, quadratic, (np.full((2, 3), -0.05), np.full((2, 3), 0.08)))
    out = _abs(x)
    bounds = taylor_range(x)
    for index in itertools.product(range(2), range(3)):
        entry, result = x[index], out[index]
        if float(bounds.lower[index]) >= 0:
            _assert_same(result, entry)
        elif float(bounds.upper[index]) <= 0:
            _assert_same(result, _neg(entry))
    _assert_contains(out, x)
    _assert_same(tmif(jnp.abs)(x), out)
    assert taylor_inclusion_registry[lax.abs_p] is _abs


@pytest.mark.parametrize("x", [POSITIVE_SLOPE, NEGATIVE_SLOPE, _model(0.0, [0.01], remainder=(-0.75, 0.75)),
                               _model(0.5, [1.0]), _model(0.2, [1.0], [[0.3]], (-0.1, 0.05))])
def test_no_wider_than_previous_rule(x):
    new, old = taylor_range(_abs(x)), taylor_range(_old_abs(x))
    assert float(new.upper) <= float(old.upper) + 1e-14
    assert float(new.upper - new.lower) <= float(old.upper - old.lower) + 1e-14
    new_width = float(_abs(x).remainder.upper - _abs(x).remainder.lower)
    old_width = float(_old_abs(x).remainder.upper - _old_abs(x).remainder.lower)
    assert new_width <= old_width + 1e-14


def test_dtype_jit_and_trace_size():
    x32 = _model(0.3, [0.5, -0.2], remainder=(-0.04, 0.07), dtype=jnp.float32)
    out = _abs(x32)
    assert all(leaf.dtype == np.float32 for leaf in _leaves(out))
    for x in (POSITIVE_SLOPE, NEGATIVE_SLOPE, _model(1.5, [0.2, 0.1])):
        _assert_same(jax.jit(_abs)(x), _abs(x))
        _assert_same(jax.jit(tmif(jnp.abs))(x), _abs(x))
    counts = [len(jax.make_jaxpr(_abs)(_model(np.zeros(4), np.full((4, n), 0.1))).jaxpr.eqns) for n in (1, 3, 12)]
    assert counts[0] == counts[1] == counts[2]


def test_interval_and_point_inputs():
    interval = _abs(Interval(jnp.array([-1.0, 2.0, -3.0, 0.0]), jnp.array([0.5, 3.0, -2.0, 0.0])))
    np.testing.assert_array_equal(np.asarray(interval.lower), [0.0, 2.0, 2.0, 0.0])
    np.testing.assert_array_equal(np.asarray(interval.upper), [1.0, 3.0, 3.0, 0.0])
    np.testing.assert_array_equal(np.asarray(_abs(jnp.array([-2.0, 0.0, 3.0]))), [2.0, 0.0, 3.0])


def test_nonfinite_ranges_are_conservative_and_finite_extremes_have_no_nan():
    unbounded = _model([0.0, np.nan], [[1.0], [1.0]], remainder=(np.array([-np.inf, 0.0]), np.array([1.0, 0.0])))
    out = _abs(unbounded)
    assert not any(np.any(np.isnan(leaf)) for leaf in _leaves(out))
    np.testing.assert_array_equal(np.asarray(out.remainder.lower), [0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(out.remainder.upper), [np.inf, np.inf])
    assert np.all(np.asarray(out.linear) == 0)
    extreme = _model(0.25e308, [1.4e308])  # finite range whose width u - l overflows
    result = _abs(extreme)
    assert not any(np.any(np.isnan(leaf)) for leaf in _leaves(result))
    for v in (-1.15e308, 0.0, 1.65e308):
        p = evaluate_polynomial(result, jnp.asarray([(v - 0.25e308) / 1.4e308]))
        assert float(p + result.remainder.lower) <= abs(v) <= float(p + result.remainder.upper)


ARCHIVES = Path(__file__).resolve().parents[2] / "tmp/npz/t_pushing/runs"


def test_offline_e1_range_aware_upper_regression():
    stage0 = ARCHIVES / "t_pushing_empty_push_range_aware_abs_stage0.npz"
    run = ARCHIVES / "t_pushing_empty_push_range_aware_abs_run.npz"
    if not (stage0.exists() and run.exists()):
        pytest.skip("optional frozen E1 range-aware-abs archives are absent")
    with np.load(stage0, allow_pickle=False) as s, np.load(run, allow_pickle=False) as r:
        g = {k: s["forced_G_" + k] for k in ("constant", "linear", "quadratic", "rem_lower", "rem_upper")}
        saved = {k: r[k] for k in r.files}
    G = TaylorModel(jnp.asarray(g["constant"]), jnp.asarray(g["linear"]), jnp.asarray(g["quadratic"]),
                    Interval(jnp.asarray(g["rem_lower"]), jnp.asarray(g["rem_upper"]))).reshape(8, 8)
    absolute = _abs(G)
    np.testing.assert_array_equal(np.asarray(absolute.linear), saved["upper_entry_linear"])
    np.testing.assert_array_equal(np.asarray(absolute.quadratic), saved["upper_entry_quadratic"])
    np.testing.assert_array_equal(np.asarray(absolute.remainder.upper), saved["upper_entry_rem_upper"])
    np.testing.assert_allclose(np.asarray(absolute.constant), saved["upper_entry_constant"], rtol=0, atol=1e-15)
    rows = np.asarray(taylor_range(_reduce_sum(absolute, axes=(1,))).upper)
    np.testing.assert_allclose(rows, saved["row_upper"], rtol=0, atol=1e-14)
    old_rows = np.asarray(taylor_range(_reduce_sum(_old_abs(G), axes=(1,))).upper)
    np.testing.assert_array_equal(old_rows, saved["direct_rows_upper"])
    lower = np.asarray(taylor_range(absolute).lower)
    assert np.all(np.isfinite(lower)) and np.all(lower <= np.asarray(taylor_range(absolute).upper))
