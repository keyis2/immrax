"""Aligned, diagnostic-only comparisons of quadratic pair product rules.

The polynomial-range inequalities in the implementation establish inclusion.
Sampled fibers here check implementation and tightness, not certification.
"""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion import PairedQuadratic, evaluate_pair, pair_range, pair_status, pqif
from immrax.inclusion.interval import Interval
from immrax.inclusion.paired_quadratic import _mul, _square_pair
from immrax.inclusion.taylor import TaylorModel

jax.config.update("jax_enable_x64", True)

STRATEGIES = ("taylor", "four_candidate", "midpoint")
TOL = 3e-11


def _model(c, a, q):
    zero = jnp.asarray(0.0)
    return TaylorModel(jnp.asarray(c), jnp.asarray(a), jnp.asarray(q),
                       Interval(zero, zero))


def _input(n, sign, rng, random=False):
    linear = rng.normal(scale=0.075, size=n) if random else np.resize([0.11, -0.08], n)
    raw = rng.normal(scale=0.025, size=(n, n)) if random else np.diag(np.resize([0.06, -0.04], n))
    quadratic = 0.5 * (raw + raw.T)
    center = _model(sign * 0.8, linear, quadratic)
    width_linear = rng.normal(scale=0.008, size=n) if random else np.resize([0.012, -0.009], n)
    diagonal = rng.uniform(0.012, 0.035, size=n) if random else np.resize([0.025, 0.018], n)
    width = _model(0.12 + np.sum(np.abs(width_linear)), width_linear,
                   np.diag(2 * diagonal))
    return PairedQuadratic(center - width, center + width)


def _sources(n, rng):
    corners = np.asarray(list(product((-1.0, 1.0), repeat=n))) if n == 2 else np.empty((0, n))
    return jnp.asarray(np.concatenate((corners, rng.uniform(-1, 1, (32, n))), axis=0))


def _fibers(value, sources):
    low, high = jax.vmap(lambda source: evaluate_pair(value, source))(sources)
    return np.asarray(low), np.asarray(high)


def _exact_product(x, y, sources):
    xl, xu = _fibers(x, sources)
    yl, yu = _fibers(y, sources)
    corners = np.stack((xl * yl, xl * yu, xu * yl, xu * yu))
    return corners.min(axis=0), corners.max(axis=0)


def _check(result, exact, sources):
    status = pair_status(result)
    for key in ("valid", "zero_remainder", "ordered", "finite"):
        assert bool(jnp.all(status[key])), (key, status)
    lower, upper = _fibers(result, sources)
    assert np.min(exact[0] - lower) >= -TOL
    assert np.min(upper - exact[1]) >= -TOL
    return float(pair_range(result).width), float(np.mean(upper - lower))


def _outcome(candidate, baseline):
    if candidate < baseline - 1e-10:
        return "win"
    if candidate > baseline + 1e-10:
        return "loss"
    return "tie"


def test_aligned_products_fixed_sign_crossing_and_random_quadratic_width():
    rng = np.random.default_rng(20260930)
    counts = {s: {metric: {key: 0 for key in ("win", "loss", "tie")}
                  for metric in ("global", "mean")}
              for s in STRATEGIES[1:]}
    for n in (2, 8):
        sources = _sources(n, rng)
        cases = ((1, 1), (-1, -1), (1, -1), (-1, 1), (0, 0))
        cases += tuple((int(rng.choice((-1, 0, 1))), int(rng.choice((-1, 0, 1))))
                       for _ in range(3))
        for index, (sx, sy) in enumerate(cases):
            x = _input(n, sx, rng, index >= 5)
            y = _input(n, sy, rng, index >= 5)
            if index == 4:
                xl, xu = _fibers(x, sources)
                yl, yu = _fibers(y, sources)
                assert np.any((xl < 0) & (xu > 0) & (yl < 0) & (yu > 0))
            exact = _exact_product(x, y, sources)
            measurements = {}
            for strategy in STRATEGIES:
                result = _mul(x, y, multiplication_strategy=strategy)
                measurements[strategy] = _check(result, exact, sources)
                if index == 0 and strategy == "midpoint":
                    default = pqif(lambda a, b: a * b)(x, y)
                    explicit = pqif(lambda a, b: a * b,
                                    multiplication_strategy="midpoint")(x, y)
                    for first, second in zip(jax.tree.leaves(default),
                                             jax.tree.leaves(explicit)):
                        np.testing.assert_allclose(first, second, atol=TOL)
            for strategy in STRATEGIES[1:]:
                for i, metric in enumerate(("global", "mean")):
                    counts[strategy][metric][
                        _outcome(measurements[strategy][i], measurements["taylor"][i])
                    ] += 1
    print("Aligned product width W/L/T vs Taylor:", counts)
    print("Sampling is regression/tightness evidence only.")


def test_specialized_square_fixed_sign_and_crossing():
    rng = np.random.default_rng(20260930)
    comparisons = []
    equation_counts = {}
    for n in (2, 8):
        sources = _sources(n, rng)
        for sign in (1, -1, 0):
            x = _input(n, sign, rng)
            lower, upper = _fibers(x, sources)
            exact_lower = np.where((lower <= 0) & (upper >= 0), 0,
                                   np.minimum(lower**2, upper**2))
            exact = exact_lower, np.maximum(lower**2, upper**2)
            specialized = _square_pair(x)
            native_metrics = _check(specialized, exact, sources)
            baseline = pqif(lambda a: a**2, multiplication_strategy="taylor")(x)
            baseline_metrics = _check(baseline, exact, sources)
            comparisons.append((n, sign, native_metrics, baseline_metrics))
            transformed = pqif(lambda a: a**2, multiplication_strategy="four_candidate")(x)
            compiled = jax.jit(_square_pair)(x)
            batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
            vmapped = jax.vmap(_square_pair)(batched)
            for result in (transformed, compiled,
                           jax.tree.map(lambda leaf: leaf[0], vmapped)):
                for a, b in zip(jax.tree.leaves(specialized), jax.tree.leaves(result)):
                    np.testing.assert_allclose(a, b, atol=TOL)
            if sign == 0:
                generic = _mul(x, x, multiplication_strategy="four_candidate")
                specialized_lower, _ = _fibers(specialized, sources)
                generic_lower, _ = _fibers(generic, sources)
                assert np.any(specialized_lower > generic_lower + 1e-4)
            jaxpr = jax.make_jaxpr(_square_pair)(x).jaxpr
            equation_counts[n, sign] = len(jaxpr.eqns)
            assert not any("scatter" in eqn.primitive.name for eqn in jaxpr.eqns)
            for eqn in jaxpr.eqns:
                for variable in eqn.outvars:
                    shape = getattr(getattr(variable, "aval", None), "shape", ())
                    assert sum(d == n for d in shape) <= 2, shape
    for sign in (1, -1, 0):
        assert equation_counts[2, sign] == equation_counts[8, sign]
    print("Square sign/native(global,mean)/Taylor(global,mean):", comparisons)


def test_eager_jit_vmap_and_fixed_rank_jaxpr_structure():
    rng = np.random.default_rng(20260930)
    equation_counts = {}
    for strategy in STRATEGIES[1:]:
        for n in (2, 8):
            x, y = _input(n, 0, rng, True), _input(n, 0, rng, True)
            rule = lambda a, b: _mul(a, b, multiplication_strategy=strategy)
            eager = rule(x, y)
            compiled = jax.jit(rule)(x, y)
            batched_x = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
            batched_y = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), y)
            vmapped = jax.vmap(rule)(batched_x, batched_y)
            for actual in (compiled, jax.tree.map(lambda leaf: leaf[0], vmapped)):
                for expected_leaf, actual_leaf in zip(jax.tree.leaves(eager), jax.tree.leaves(actual)):
                    np.testing.assert_allclose(actual_leaf, expected_leaf, atol=TOL)
            jaxpr = jax.make_jaxpr(rule)(x, y).jaxpr
            equation_counts[n, strategy] = len(jaxpr.eqns)
            assert not any("scatter" in eqn.primitive.name for eqn in jaxpr.eqns)
            for eqn in jaxpr.eqns:
                for variable in eqn.outvars:
                    shape = getattr(getattr(variable, "aval", None), "shape", ())
                    assert sum(d == n for d in shape) <= 2, shape
            assert eager.lower.quadratic.shape == (n, n)
            assert eager.upper.quadratic.shape == (n, n)
            transformed = pqif(lambda a, b: a * b,
                               multiplication_strategy=strategy)(x, y)
            for expected_leaf, actual_leaf in zip(jax.tree.leaves(eager), jax.tree.leaves(transformed)):
                np.testing.assert_allclose(actual_leaf, expected_leaf, atol=TOL)
            if n == 2:
                nested = pqif(lambda a, b: jax.jit(lambda u, v: u * v)(a, b),
                              multiplication_strategy=strategy)(x, y)
                for expected_leaf, actual_leaf in zip(jax.tree.leaves(eager), jax.tree.leaves(nested)):
                    np.testing.assert_allclose(actual_leaf, expected_leaf, atol=TOL)
        assert equation_counts[2, strategy] == equation_counts[8, strategy]
    print("Source-stable Jaxpr equation counts:", equation_counts)


def test_native_invalid_order_and_nonfinite_input_remain_visible():
    rng = np.random.default_rng(20260930)
    good = _input(2, 0, rng)
    bad_order = PairedQuadratic(good.upper, good.lower)
    bad_finite = PairedQuadratic(
        _model(np.nan, np.zeros(2), np.zeros((2, 2))), good.upper,
    )
    for strategy in STRATEGIES[1:]:
        for bad in (bad_order, bad_finite):
            result = _mul(bad, good, multiplication_strategy=strategy)
            assert not bool(jnp.all(pair_status(result)["valid"]))
            assert bool(jnp.any(jnp.isnan(result.lower.constant)))
    try:
        pqif(lambda x: x, multiplication_strategy="unsupported")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown static multiplication strategy was accepted")
