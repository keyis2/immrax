"""Supported paired-quadratic representation, nonlinear rules and transform."""

import itertools
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from immrax.inclusion.interval import Interval
from immrax.inclusion.paired_quadratic import (
    PairedQuadratic,
    constant_pair,
    evaluate_pair,
    normalized_pair_seed,
    pair_divide,
    pair_abs,
    pair_from_interval,
    pair_integer_pow,
    pair_max,
    pair_min,
    pair_multiply,
    pair_range,
    pair_reciprocal,
    pair_sqrt,
    pair_square,
    pair_trig,
)
from immrax.inclusion.paired_quadratic_transform import (
    UnsupportedPairPrimitive,
    pair_primitive_rules,
    pqif,
)
from immrax.inclusion.quadratic_polynomial import (
    QuadraticPolynomial,
    constant_quadratic,
    evaluate_quadratic,
    normalized_quadratic_seed,
    quadratic_range,
)

jax.config.update("jax_enable_x64", True)


def _coefficients(shape, m, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape), rng.normal(size=shape + (m,)),
            rng.normal(size=shape + (m, m)))  # Q deliberately nonsymmetric.


def _polynomial(shape=(2, 3), m=3, seed=0):
    return QuadraticPolynomial(*_coefficients(shape, m, seed))


def _pair(shape=(2, 3), m=3, seed=0):
    """Ordered pair L = C - W, U = C + W with W a positive constant plus a PSD form."""
    center = _polynomial(shape, m, seed)
    rng = np.random.default_rng(seed + 100)
    factor = rng.normal(size=shape + (m, m))
    gram = np.einsum("...ki,...kj->...ij", factor, factor)
    width = QuadraticPolynomial(rng.uniform(0.1, 0.5, size=shape), np.zeros(shape + (m,)), gram)
    return PairedQuadratic(center - width, center + width)


def _direct(constant, linear, quadratic, xi):
    return constant + linear @ xi + 0.5 * np.einsum("...ij,i,j->...", quadratic, xi, xi)


def _grid(m, points=7):
    axis = np.linspace(-1.0, 1.0, points)
    return np.array(list(itertools.product(axis, repeat=m)))


def _assert_same(first, second):
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_array_equal(a, b)


def test_construction_validates_shape_and_dtype_and_has_no_remainder():
    c, l, q = _coefficients((2,), 3, 0)
    polynomial = QuadraticPolynomial(c, l, q)
    assert not hasattr(polynomial, "remainder")
    assert (polynomial.shape, polynomial.ndim, polynomial.size, polynomial.source_size,
            polynomial.dtype) == ((2,), 1, 2, 3, jnp.float64)
    with pytest.raises(ValueError, match="linear shape"):
        QuadraticPolynomial(c, l[None], q)
    with pytest.raises(ValueError, match="quadratic shape"):
        QuadraticPolynomial(c, l, q[..., :2])
    with pytest.raises(ValueError, match="share one dtype"):
        QuadraticPolynomial(c.astype(np.float32), l, q)
    with pytest.raises(TypeError, match="floating"):
        QuadraticPolynomial(np.zeros(2, int), np.zeros((2, 3), int), np.zeros((2, 3, 3), int))

    pair = PairedQuadratic(polynomial, polynomial)
    assert not hasattr(pair, "remainder") and pair.valid.shape == (2,)
    with pytest.raises(TypeError):
        PairedQuadratic(polynomial.constant, polynomial.constant)
    with pytest.raises(ValueError, match="value/source"):
        PairedQuadratic(polynomial, _polynomial((2,), 4))
    with pytest.raises(ValueError, match="value/source"):
        PairedQuadratic(polynomial, polynomial[:1])
    cast = QuadraticPolynomial(c.astype(np.float32), l.astype(np.float32), q.astype(np.float32))
    with pytest.raises(ValueError, match="dtype"):
        PairedQuadratic(polynomial, cast)

    # Exact layers refuse relaxation-requiring operands.
    for bad in (lambda: polynomial * polynomial, lambda: polynomial / polynomial,
                lambda: polynomial + Interval(c, c), lambda: pair * Interval(c, c),
                lambda: pair + Interval(c, c), lambda: pair + polynomial):
        with pytest.raises(TypeError):
            bad()


@pytest.mark.parametrize("shape, m", [((), 1), ((), 3), ((4,), 2), ((2, 3), 3)])
def test_point_evaluation_matches_direct_formula(shape, m):
    c, l, q = _coefficients(shape, m, 1)
    polynomial = QuadraticPolynomial(c, l, q)
    for xi in np.random.default_rng(2).uniform(-1, 1, size=(5, m)):
        np.testing.assert_allclose(evaluate_quadratic(polynomial, xi), _direct(c, l, q, xi),
                                   rtol=1e-14, atol=1e-14)


def test_termwise_range_contains_dense_design_and_is_tight_on_simple_terms():
    for shape, m, seed in (((), 2, 3), ((3,), 3, 4), ((2, 2), 4, 5)):
        polynomial = _polynomial(shape, m, seed)
        bounds = quadratic_range(polynomial)
        values = jax.vmap(lambda xi: evaluate_quadratic(polynomial, xi))(_grid(m))
        assert bool(jnp.all(values >= bounds.lower[None] - 1e-12))
        assert bool(jnp.all(values <= bounds.upper[None] + 1e-12))

    zero = np.zeros(2)
    # xi_0 xi_1 split across a nonsymmetric Q: range [-1, 1]; xi_0^2: range [0, 1].
    cross = quadratic_range(QuadraticPolynomial(0.0, zero, np.array([[0.0, 1.5], [0.5, 0.0]])))
    square = quadratic_range(QuadraticPolynomial(0.0, zero, np.array([[2.0, 0.0], [0.0, 0.0]])))
    np.testing.assert_allclose([cross.lower, cross.upper], [-1.0, 1.0])
    np.testing.assert_allclose([square.lower, square.upper], [0.0, 1.0])

    pair = _pair((2,), 3, 6)
    hull = pair_range(pair)
    lower, upper = jax.vmap(lambda xi: evaluate_pair(pair, xi))(_grid(3))
    assert bool(jnp.all(lower >= hull.lower - 1e-12) & jnp.all(upper <= hull.upper + 1e-12))
    assert bool(jnp.all(lower <= upper))


def test_normalized_seed_and_constant_pair():
    box_lower, box_upper = jnp.array([-0.2, 1.0, 3.0]), jnp.array([0.4, 1.0, 5.0])
    seed = normalized_pair_seed(box_lower, box_upper)
    for xi in _grid(3, 3):
        lower, upper = evaluate_pair(seed, xi)
        expected = 0.5 * (box_lower + box_upper) + 0.5 * (box_upper - box_lower) * xi
        np.testing.assert_allclose(lower, expected)
        np.testing.assert_array_equal(lower, upper)
    hull = pair_range(seed)
    np.testing.assert_allclose(hull.lower, box_lower)
    np.testing.assert_allclose(hull.upper, box_upper)
    assert bool(jnp.all(jnp.isnan(normalized_quadratic_seed(box_upper, box_lower).constant)))
    with pytest.raises(ValueError):
        normalized_quadratic_seed(jnp.zeros((2, 2)), jnp.ones((2, 2)))

    constant = constant_pair(jnp.array([[1.5, -2.0]]), 3)
    assert constant.shape == (1, 2) and constant.source_size == 3
    hull = pair_range(constant)
    np.testing.assert_array_equal(hull.lower, [[1.5, -2.0]])
    np.testing.assert_array_equal(hull.upper, [[1.5, -2.0]])
    assert constant_quadratic(2, 0).dtype == jnp.float64


def test_exact_linear_operations_and_negative_scale_swap():
    x, y = _pair((2, 3), 3, 7), _pair((2, 3), 3, 8)
    offset = jnp.array([0.5, -1.0, 2.0])
    scale = jnp.array([[2.0, -0.5, 0.0], [-3.0, 1.0, -1.0]])
    for xi in _grid(3, 3):
        (xl, xu), (yl, yu) = evaluate_pair(x, xi), evaluate_pair(y, xi)
        cases = {
            "add": (x + y, xl + yl, xu + yu),
            "sub": (x - y, xl - yu, xu - yl),
            "neg": (-x, -xu, -xl),
            "shift": (offset + x - offset * 2, xl - offset, xu - offset),
            "rsub": (offset - x, offset - xu, offset - xl),
            "negative scalar": (-2.0 * x, -2.0 * xu, -2.0 * xl),
            "mixed array": (x * scale, jnp.minimum(xl * scale, xu * scale),
                            jnp.maximum(xl * scale, xu * scale)),
        }
        for name, (value, expected_lower, expected_upper) in cases.items():
            lower, upper = evaluate_pair(value, xi)
            np.testing.assert_allclose(lower, expected_lower, atol=1e-12, err_msg=name)
            np.testing.assert_allclose(upper, expected_upper, atol=1e-12, err_msg=name)
            assert bool(jnp.all(lower <= upper)), name

    polynomial = _polynomial((2, 3), 3, 9)
    xi = jnp.array([0.3, -0.7, 0.1])
    np.testing.assert_allclose(evaluate_quadratic(polynomial / scale.at[0, 2].set(4.0), xi),
                               evaluate_quadratic(polynomial, xi) / scale.at[0, 2].set(4.0))
    np.testing.assert_allclose(evaluate_quadratic(np.float64(3.0) * polynomial - 1.0, xi),
                               3.0 * evaluate_quadratic(polynomial, xi) - 1.0)
    assert (np.array([1.0, 2.0, 3.0]) * polynomial).shape == (2, 3)
    assert (x * jnp.array([1.0, jnp.nan, -1.0])).valid.tolist() == [[True, False, True]] * 2
    assert ((x * 2.0) + jnp.zeros((4, 2, 3))).valid.shape == (4, 2, 3)


def test_shape_operations_jit_vmap_and_pytree_round_trip():
    pair = _pair((2, 3), 3, 10)
    xi = jnp.array([0.2, -0.4, 0.9])
    lower, upper = evaluate_pair(pair, xi)
    for value, (expected_lower, expected_upper), shape in (
        (pair[1], (lower[1], upper[1]), (3,)),
        (pair[:, 1:], (lower[:, 1:], upper[:, 1:]), (2, 2)),
        (pair.reshape(6), (lower.reshape(6), upper.reshape(6)), (6,)),
        (pair.reshape((3, 2)), (lower.reshape(3, 2), upper.reshape(3, 2)), (3, 2)),
        (pair.T, (lower.T, upper.T), (3, 2)),
        (pair.reshape(1, 2, 3).transpose(2, 0, 1),
         (lower.reshape(1, 2, 3).transpose(2, 0, 1), upper.reshape(1, 2, 3).transpose(2, 0, 1)),
         (3, 1, 2)),
    ):
        assert value.shape == value.valid.shape == shape and value.source_size == 3
        result = evaluate_pair(value, xi)
        np.testing.assert_allclose(result[0], expected_lower)
        np.testing.assert_allclose(result[1], expected_upper)

    leaves, treedef = jax.tree.flatten(pair)
    assert len(leaves) == 7
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, PairedQuadratic) and isinstance(rebuilt.lower, QuadraticPolynomial)
    for first, second in zip(jax.tree.leaves(rebuilt), leaves):
        np.testing.assert_array_equal(first, second)

    jitted = jax.jit(lambda p: (-0.5 * p + p.T.T)[0])(pair)
    assert isinstance(jitted, PairedQuadratic)
    expected = (-0.5 * pair + pair)[0]
    for first, second in zip(jax.tree.leaves(jitted), jax.tree.leaves(expected)):
        np.testing.assert_allclose(first, second)
    jit_range = jax.jit(pair_range)(pair)
    np.testing.assert_allclose(jit_range.lower, pair_range(pair).lower)

    sources = jnp.asarray(_grid(3, 3))
    batch_lower, batch_upper = jax.jit(jax.vmap(evaluate_pair, in_axes=(None, 0)))(pair, sources)
    assert batch_lower.shape == (27, 2, 3)
    np.testing.assert_allclose(batch_lower[5], evaluate_pair(pair, sources[5])[0])
    np.testing.assert_allclose(batch_upper[5], evaluate_pair(pair, sources[5])[1])

    # Outer vmap over a stack of pairs: leaves gain a leading batch axis.
    stack = jax.tree.map(lambda *xs: jnp.stack(xs), *(_pair((2,), 3, s) for s in range(4)))
    assert isinstance(stack, PairedQuadratic) and stack.shape == (4, 2)
    batched = jax.vmap(lambda p: pair_range(2.0 - 3.0 * p))(stack)
    single = pair_range(2.0 - 3.0 * _pair((2,), 3, 2))
    np.testing.assert_allclose(batched.lower[2], single.lower)
    np.testing.assert_allclose(batched.upper[2], single.upper)
    built = jax.vmap(lambda lo, hi: normalized_pair_seed(lo, hi))(-jnp.ones((4, 3)), jnp.ones((4, 3)))
    assert built.shape == (4, 3) and built.lower.quadratic.shape == (4, 3, 3, 3)


def test_affine_singleton_product_and_point_division_are_exact():
    """Products of affine exact singletons have no omitted terms; scaling is coefficientwise."""
    seed = normalized_pair_seed(jnp.array([-0.5, 1.0, 2.0]), jnp.array([1.5, 3.0, 2.5]))
    p, q = seed[0] + 0.25 * seed[1], seed[2] - seed[1]
    product = p * q
    lp, lq = p.lower, q.lower
    expected = QuadraticPolynomial(lp.constant * lq.constant, lp.constant * lq.linear + lq.constant * lp.linear,
                                   jnp.outer(lp.linear, lq.linear) + jnp.outer(lq.linear, lp.linear))
    for end in (product.lower, product.upper):
        np.testing.assert_allclose(end.constant, expected.constant, rtol=1e-15)
        np.testing.assert_allclose(end.linear, expected.linear, rtol=1e-15, atol=1e-16)
        np.testing.assert_allclose(end.quadratic, expected.quadratic, rtol=1e-15, atol=1e-16)
    x = _pair((2, 3), 3, 11)
    divided = x.lower / 3.0
    np.testing.assert_array_equal(divided.quadratic, x.lower.quadratic / 3.0)
    _assert_same(x / -4.0, x * jnp.reciprocal(-4.0))


# ---------------------------------------------------------------- nonlinear core

def _signed_pair(sign, shape=(2, 3), seed=20):
    """Fixed-sign pair: a small ordered pair shifted away from zero."""
    pair = sign * (0.05 * _pair(shape, 3, seed) + 2.0)
    bounds = pair_range(pair)
    assert bool(jnp.all(bounds.lower > 0) if sign > 0 else jnp.all(bounds.upper < 0))
    return pair


def _margin(result, function, *pairs, points=7):
    """Least sampled containment margin over the result's source box."""
    grid = jnp.asarray(_grid(result.source_size, points))
    lower, upper = jax.vmap(lambda xi: evaluate_pair(result, xi))(grid)
    ends = [jax.vmap(lambda xi, p=p: evaluate_pair(p, xi))(grid) for p in pairs]
    margin = np.inf
    for ts in itertools.product((0.0, 0.25, 0.5, 1.0), repeat=len(pairs)):
        value = function(*(a + t * (b - a) for (a, b), t in zip(ends, ts)))
        margin = min(margin, float(jnp.min(value - lower)), float(jnp.min(upper - value)))
    return margin


def test_nonlinear_rules_contain_dense_samples():
    x, y = _pair((2, 3), 3, 13), _pair((2, 3), 3, 14)
    positive, negative = _signed_pair(1.0, seed=22), _signed_pair(-1.0, seed=23)
    assert bool(jnp.any(pair_range(x).lower < 0) & jnp.any(pair_range(x).upper > 0))
    checks = (
        (x * y, lambda a, b: a * b, x, y),
        (pair_square(x), lambda a: a * a, x),
        (pair_square(negative), lambda a: a * a, negative),
        (pair_reciprocal(positive), lambda a: 1 / a, positive),
        (pair_reciprocal(negative), lambda a: 1 / a, negative),
        (x / positive, lambda a, b: a / b, x, positive),
        (x / negative, lambda a, b: a / b, x, negative),
        (3.0 / negative, lambda a: 3.0 / a, negative),
    )
    for result, function, *pairs in checks:
        assert bool(jnp.all(result.valid))
        assert _margin(result, function, *pairs) >= -1e-12


def _crossing_denominator(shape=(2, 3), element=(1, 0), seed=24):
    """Positive pair with one element shifted so its range crosses zero."""
    positive = _signed_pair(1.0, shape, seed)
    bounds = pair_range(positive)
    crossing = np.zeros(shape, bool)
    crossing[element] = True
    denominator = positive - jnp.where(crossing, 0.5 * (bounds.lower + bounds.upper), 0.0)
    assert float(pair_range(denominator).lower[element]) < 0 < float(pair_range(denominator).upper[element])
    return denominator, crossing


def test_zero_crossing_reciprocal_and_division_are_rejected_per_element():
    denominator, crossing = _crossing_denominator()
    x = _pair((2, 3), 3, 15)
    for result in (pair_reciprocal(denominator), x / denominator, 1.0 / denominator):
        np.testing.assert_array_equal(result.valid, ~crossing)
        assert bool(jnp.all(jnp.isnan(result.lower.linear[crossing])) & jnp.all(jnp.isnan(result.upper.constant[crossing])))
        assert bool(jnp.all(jnp.isfinite(result.lower.quadratic[~crossing])))


def _is_singleton(value):
    return all(bool(jnp.all(a == b)) for a, b in zip(jax.tree.leaves(value.lower), jax.tree.leaves(value.upper)))


def test_exact_singleton_and_point_factor_behavior():
    box_lower, box_upper = jnp.array([-0.5, 1.0, 2.0]), jnp.array([1.5, 3.0, 2.5])
    seed = normalized_pair_seed(box_lower, box_upper)
    x = _pair((3,), 3, 16)
    for value in (-3.0 * seed, seed / -4.0, seed / constant_pair(jnp.full(3, -4.0), 3), seed * seed):
        assert _is_singleton(value) and bool(jnp.all(value.valid))
    # A constant singleton pair denominator takes the same one-reciprocal scaling.
    for first, second in zip(jax.tree.leaves(seed / constant_pair(jnp.full(3, -4.0), 3)),
                             jax.tree.leaves(seed / -4.0)):
        np.testing.assert_array_equal(first, second)
    # The affine seed has no omitted degree-3/4 terms: x_i^2 is reproduced exactly.
    xi = jnp.array([0.3, -1.0, 0.7])
    physical = 0.5 * (box_lower + box_upper) + 0.5 * (box_upper - box_lower) * xi
    np.testing.assert_allclose(evaluate_pair(seed * seed, xi)[0], physical**2, rtol=1e-14)
    # A negative point divisor swaps endpoints exactly like scaling by its reciprocal.
    for first, second in zip(jax.tree.leaves(x / -0.5), jax.tree.leaves(-2.0 * x)):
        np.testing.assert_array_equal(first, second)
    lower, upper = evaluate_pair(x / -0.5, xi)
    original = evaluate_pair(x, xi)
    np.testing.assert_array_equal(lower, -2.0 * original[1])
    np.testing.assert_array_equal(upper, -2.0 * original[0])
    with pytest.raises(NotImplementedError):
        pair_integer_pow(x, 3)


_OPERATIONS = {
    "mul": lambda a, b: pair_multiply(a, b),
    "square": lambda a, b: pair_square(a),
    "pow0": lambda a, b: pair_integer_pow(a, 0),
    "reciprocal": lambda a, b: pair_reciprocal(b),
    "divide": lambda a, b: pair_divide(a, b),
    "point divide": lambda a, b: pair_divide(a, constant_pair(jnp.full(a.shape, -4.0), 3)),
}


@pytest.mark.parametrize("name", list(_OPERATIONS))
def test_nonlinear_rules_under_jit_and_outer_vmap(name):
    operation = _OPERATIONS[name]
    xs = [_pair((3,), 3, seed) for seed in (30, 31, 32)]
    ds = [_signed_pair(1.0, (3,), seed) for seed in (33, 34, 35)]
    eager = operation(xs[0], ds[0])
    jitted = jax.jit(operation)(xs[0], ds[0])
    for first, second in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager)):
        np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-14)
    stack = lambda values: jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)
    batched = jax.jit(jax.vmap(operation))(stack(xs), stack(ds))
    assert batched.shape == (3, 3)
    for index in range(3):
        single = operation(xs[index], ds[index])
        for first, second in zip(jax.tree.leaves(batched[index]), jax.tree.leaves(single)):
            np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-14)


def test_invalid_and_nonfinite_inputs_propagate_per_element():
    x, d = _signed_pair(1.0, seed=26), _signed_pair(1.0, seed=25)
    flagged = PairedQuadratic(x.lower, x.upper, jnp.ones((2, 3), bool).at[0, 1].set(False))
    nonfinite = PairedQuadratic(
        QuadraticPolynomial(x.lower.constant.at[1, 2].set(jnp.nan), x.lower.linear, x.lower.quadratic),
        x.upper)
    for value, expected_valid in ((flagged, np.asarray(flagged.valid)),
                                  (nonfinite, np.ones((2, 3), bool) & ~(np.arange(6) == 5).reshape(2, 3))):
        for native in (value * d, pair_square(value), pair_reciprocal(value), d / value):
            np.testing.assert_array_equal(native.valid, expected_valid)
            assert bool(jnp.all(jnp.isnan(native.lower.constant[~expected_valid])))
            assert bool(jnp.all(jnp.isfinite(native.upper.linear[expected_valid])))


# ---------------------------------------------------------------- transform

def _vector_function(x, y):
    z = x[1:] * y[:2] + x[0] / y[2]
    w = jnp.where(y[0] > 0.0, z, -z) ** 2
    v = jnp.concatenate([w, x[jnp.array([0, 2])]])
    return lax.convert_element_type(v.reshape(2, 2)[1], jnp.float64) - 0.5 * x[2] / y[:2]


def _matrix_function(a, b, p):
    return jnp.sum(a @ b, axis=1) + jnp.sum((p @ a.T).T, axis=1)


def _control_function(x, y):
    out = jnp.copy(x)
    for test in (x < y, x <= y, x > y, x >= y, x == y, x != y):
        out = jnp.where(test, out, y)
    return jnp.broadcast_to(out, (2, 3)) - y


def _dynamic_index_function(x):
    return x[-1] * x


def _nonsmooth_function(x, y):
    a = jnp.sqrt(y) + jnp.sin(x) * jnp.cos(y) - jnp.abs(x)
    b = jnp.minimum(jnp.maximum(a, x), y)
    cond = ((x > 0.0) & ~(y < 0.0)) | (x < -5.0)
    keep = jnp.all(y > 0.0) | jnp.any(x > 1e3)
    c = jnp.where(cond & keep, a, b)
    return c.at[jnp.array([0, 2])].set(b[:2]).at[jnp.array([1, 1])].add(a[:2])


def _derivative_function(x):
    """jacfwd accumulates tangents with add_any; jnp.split lowers to split."""
    jacobian = jax.jacfwd(lambda w: w * w + jnp.sin(w) * w)(x)
    first, second = jnp.split(jacobian, [1], axis=1)
    return jnp.concatenate([second, first], axis=1)


def _mixed_function(x, d, p):
    q = x / d + 1.0
    return jnp.sum(q, axis=0), jnp.sum(q, axis=1), p @ q.T, q


def _vector_inputs(seed=40):
    return _pair((3,), 3, seed), _signed_pair(1.0, (3,), seed + 1)


def _matrix_inputs(seed=42):
    return (_pair((2, 3), 3, seed), _pair((3, 2), 3, seed + 1),
            jnp.asarray(np.random.default_rng(seed).normal(size=(3, 3))))


def _transform_margin(output, function, pairs, points=(), draws=16):
    """Least sampled margin; inputs vary independently per component."""
    grid = jnp.asarray(_grid(3, 5))
    lower, upper = jax.vmap(lambda xi: evaluate_pair(output, xi))(grid)
    ends = [jax.vmap(lambda xi, p=p: evaluate_pair(p, xi))(grid) for p in pairs]
    rng = np.random.default_rng(0)
    margin = np.inf
    for draw in range(draws):
        ts = [rng.integers(0, 2, p.shape) if draw % 2 else rng.uniform(size=p.shape) for p in pairs]
        concrete = [a + t * (b - a) for (a, b), t in zip(ends, ts)]
        value = jax.vmap(lambda *xs: function(*xs, *points))(*concrete)
        margin = min(margin, float(jnp.min(value - lower)), float(jnp.min(upper - value)))
    return margin


def _primitives(jaxpr):
    names = set()
    for equation in jaxpr.eqns:
        names.add(equation.primitive)
        if "jaxpr" in equation.params:
            names |= _primitives(equation.params["jaxpr"].jaxpr)
    return names


def test_transform_fixtures_exercise_every_registered_primitive():
    x, y = _vector_inputs()
    a, b, p = _matrix_inputs()
    point = lambda v: v.lower.constant
    traced = set()
    for function, args in ((_vector_function, (x, y)), (_matrix_function, (a, b, p)),
                           (_control_function, (x, y)), (_dynamic_index_function, (x,)),
                           (_mixed_function, (a, a, p)), (_nonsmooth_function, (x, y)),
                           (_derivative_function, (x,))):
        args = [point(v) if isinstance(v, PairedQuadratic) else v for v in args]
        traced |= _primitives(jax.make_jaxpr(function)(*args).jaxpr)
    assert set(pair_primitive_rules) <= traced, set(pair_primitive_rules) - traced
    assert any(p.name == "jit" for p in traced)  # handled by the interpreter itself


def test_pqif_vector_function_contains_samples():
    x, y = _vector_inputs()
    native = pqif(_vector_function)(x, y)
    assert bool(jnp.all(native.valid))
    assert _transform_margin(native, _vector_function, (x, y)) >= -1e-12
    # Dynamic indexing x[-1] must equal the static x[2] path.
    for first, second in zip(jax.tree.leaves(pqif(_dynamic_index_function)(x)),
                             jax.tree.leaves(pqif(lambda v: v[2] * v)(x))):
        np.testing.assert_array_equal(first, second)


def test_pqif_matrix_function_contains_samples_and_scales_point_matrices_exactly():
    a, b, p = _matrix_inputs()
    native = pqif(_matrix_function)(a, b, p)
    assert bool(jnp.all(native.valid))
    # A point matrix times a pair is an exact signed scaling: a singleton stays a singleton.
    singleton = normalized_pair_seed(-jnp.ones(3), jnp.ones(3))
    image = pqif(lambda v: p @ v)(singleton)
    _assert_same(image.lower, image.upper)
    assert _transform_margin(native, _matrix_function, (a, b), (p,)) >= -1e-11


@pytest.mark.parametrize("name", ["vector", "matrix"])
def test_pqif_under_jit_and_outer_vmap(name):
    function, make = {"vector": (_vector_function, _vector_inputs),
                      "matrix": (_matrix_function, _matrix_inputs)}[name]
    transformed = pqif(function)
    inputs = [make(seed) for seed in (50, 52, 54)]
    eager = [transformed(*args) for args in inputs]
    for first, second in zip(jax.tree.leaves(jax.jit(transformed)(*inputs[0])), jax.tree.leaves(eager[0])):
        np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-13)
    stacked = [jax.tree.map(lambda *leaves: jnp.stack(leaves), *(args[i] for args in inputs))
               for i in range(len(inputs[0]))]
    batched = jax.jit(jax.vmap(transformed))(*stacked)
    assert isinstance(batched, PairedQuadratic) and batched.shape[0] == 3
    for index in range(3):
        for first, second in zip(jax.tree.leaves(batched[index]), jax.tree.leaves(eager[index])):
            np.testing.assert_allclose(first, second, rtol=1e-13, atol=1e-13)


def test_pqif_comparisons_select_decided_branches_and_hull_undecided():
    x, _ = _vector_inputs()
    y = x + 100.0  # Every comparison against x is decided.
    result = pqif(_control_function)(x, y)
    expected = pqif(lambda v, w: jnp.broadcast_to(w, (2, 3)) - w)(x, y)
    for first, second in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(first, second)
    assert bool(jnp.all(result.valid))
    overlapping = pqif(lambda v, w: jnp.where(v > w, v, w))(x, x + 0.1)
    assert bool(jnp.all(overlapping.valid))
    assert _transform_margin(overlapping, lambda v, w: jnp.where(v > w, v, w), (x, x + 0.1)) >= -1e-12


def test_pqif_interval_input_enters_as_constant_pair():
    x = _pair((3,), 3, 60)
    function = lambda v, c: v * c + c
    interval = Interval(jnp.array([0.5, -1.0, 2.0]), jnp.array([0.75, -0.5, 2.5]))
    native = pqif(function)(x, interval)
    explicit = pqif(function)(x, pair_from_interval(interval, 3))
    for first, second in zip(jax.tree.leaves(native), jax.tree.leaves(explicit)):
        np.testing.assert_array_equal(first, second)
    assert _transform_margin(native, function, (x, pair_from_interval(interval, 3))) >= -1e-12
    reversed_interval = Interval(interval.lower, interval.upper.at[2].set(1.0))
    np.testing.assert_array_equal(pqif(function)(x, reversed_interval).valid, [True, True, False])


def test_pqif_invalidity_reaches_only_dependent_outputs():
    denominator, crossing = _crossing_denominator()
    x = _pair((2, 3), 3, 15)
    p = jnp.asarray(np.random.default_rng(1).normal(size=(4, 3)))
    columns, rows, dot, q = pqif(_mixed_function)(x, denominator, p)
    for value, expected in ((q, ~crossing), (columns, [False, True, True]), (rows, [True, False]),
                            (dot, np.tile([True, False], (4, 1)))):
        expected = np.asarray(expected)
        np.testing.assert_array_equal(value.valid, expected)
        assert bool(jnp.all(jnp.isfinite(value.lower.quadratic[expected])))
        assert bool(jnp.all(jnp.isnan(value.upper.constant[~expected])))


def test_pqif_unsupported_primitive_raises_without_fallback():
    x, _ = _vector_inputs()
    assert lax.exp_p not in pair_primitive_rules
    with pytest.raises(UnsupportedPairPrimitive, match="unsupported .*: exp$"):
        pqif(lambda v: jnp.exp(v) + v)(x)
    # An unsupported primitive outside the live output slice is not required.
    dead = pqif(lambda v: (jnp.exp(v), 2.0 * v)[1])(x)
    np.testing.assert_array_equal(dead.upper.constant, (2.0 * x).upper.constant)
    with pytest.raises(NotImplementedError):
        pqif(lambda v: v**3)(x)


def _unary_cases():
    x, y = 0.1 * _pair((2, 3), 3, 11), 0.1 * _pair((2, 3), 3, 12)
    positive = _signed_pair(1.0, seed=20)
    seed = normalized_pair_seed(jnp.array([-0.5, 1.4, 3.0]), jnp.array([1.5, 1.8, 3.2]))
    wide = 3.0 * _pair((2, 3), 3, 13)
    neutral = jnp.array([[-jnp.inf, 0.1, -jnp.inf], [0.0, -jnp.inf, 0.2]])
    cases = [("sqrt", pair_sqrt(positive), jnp.sqrt, (positive,)),
             ("abs", pair_abs(x), jnp.abs, (x,)),
             ("max", pair_max(x, y), jnp.maximum, (x, y)),
             ("min", pair_min(x, y), jnp.minimum, (x, y)),
             ("max -inf", pair_max(x, neutral), lambda a: jnp.maximum(a, neutral), (x,))]
    for kind, function in (("sin", jnp.sin), ("cos", jnp.cos)):
        for name, value in (("monotone", x), ("critical singleton", seed), ("critical hull", wide)):
            cases.append((f"{kind} {name}", pair_trig(kind, value), function, (value,)))
    return cases


def test_unary_and_nonsmooth_rules_contain_samples():
    for name, result, function, operands in _unary_cases():
        assert bool(jnp.all(result.valid)), name
        assert _margin(result, function, *operands) >= -1e-12, name
    # max(pair, -inf) is the pair itself; a non-singleton range across a
    # critical point keeps the constant trig hull [-1, 1].
    x = 0.1 * _pair((2, 3), 3, 11)
    neutral = pair_max(x, jnp.full((2, 3), -jnp.inf))
    _assert_same((neutral.lower, neutral.upper), (x.lower, x.upper))
    hull = pair_trig("sin", 3.0 * _pair((2, 3), 3, 13))
    critical = ~((jnp.floor((pair_range(3.0 * _pair((2, 3), 3, 13)).lower - jnp.pi / 2) / jnp.pi))
                 == jnp.floor((pair_range(3.0 * _pair((2, 3), 3, 13)).upper - jnp.pi / 2) / jnp.pi))
    assert bool(jnp.any(critical))
    np.testing.assert_array_equal(hull.lower.constant[critical], -1.0)
    np.testing.assert_array_equal(hull.upper.constant[critical], 1.0)


def test_sqrt_rejects_negative_domain_per_element():
    denominator, crossing = _crossing_denominator()
    result = pair_sqrt(denominator)
    np.testing.assert_array_equal(result.valid, ~crossing)
    assert bool(jnp.all(jnp.isnan(result.upper.constant[crossing])))


def test_trig_signed_curvature_retains_one_sided_small_angle_quadratics():
    radius = 1e-3
    x = normalized_pair_seed(jnp.array([-radius]), jnp.array([radius]))[0]
    cosine, sine = pair_trig("cos", x), pair_trig("sin", x)
    # Cosine is concave on this whole interval. Its lower quadratic retains
    # -delta**2/2 rather than adding a symmetric curvature remainder.
    np.testing.assert_allclose(cosine.lower.quadratic, [[-radius**2]], rtol=1e-14)
    np.testing.assert_allclose(cosine.upper.quadratic, [[-np.cos(radius)*radius**2]], rtol=1e-14)
    assert float(quadratic_range(.5*(cosine.upper-cosine.lower)).upper) < 3e-13
    assert float(quadratic_range(.5*(sine.upper-sine.lower)).upper) < 6e-10
    for kind, function, result in (("cos", jnp.cos, cosine), ("sin", jnp.sin, sine)):
        assert _margin(result, function, x) >= -1e-14, kind


def test_trig_quadratic_input_signed_remainder_and_nonsingleton_domains():
    # Both negative-curvature cases have a genuinely quadratic delta, whose
    # square has distinct degree-reduced endpoints. Using square_upper for
    # the negative upper remainder would fail containment near xi=0.
    p = QuadraticPolynomial(jnp.array(.7), jnp.array([.2]), jnp.array([[.3]]))
    exact = PairedQuadratic(p, p)
    for kind, function in (("sin", jnp.sin), ("cos", jnp.cos)):
        for value in (exact, PairedQuadratic(p-.01, p+.01), -exact):
            result = pair_trig(kind, value)
            assert bool(result.valid)
            assert _margin(result, function, value) >= -1e-14
    # Cross an interior maximum/minimum, and cover more than a full period.
    for center, radius in ((np.pi/2, .3), (0., .2), (2., 4.)):
        value = normalized_pair_seed(jnp.array([center-radius]), jnp.array([center+radius]))[0]
        for kind, function in (("sin", jnp.sin), ("cos", jnp.cos)):
            assert _margin(pair_trig(kind, value), function, value) >= -1e-14
    crossing = PairedQuadratic(p-.8, p+.8)
    hull = pair_trig("cos", crossing)
    np.testing.assert_array_equal(hull.lower.constant, -1.)
    np.testing.assert_array_equal(hull.upper.constant, 1.)
    np.testing.assert_array_equal(hull.lower.linear, 0.)
    np.testing.assert_array_equal(hull.upper.quadratic, 0.)


def test_trig_constant_invalid_and_jit_vmap_contract():
    for kind, function in (("sin", jnp.sin), ("cos", jnp.cos)):
        constant = pair_trig(kind, constant_pair(jnp.array([0., np.pi/2]), 1))
        np.testing.assert_array_equal(constant.lower.constant, function(jnp.array([0., np.pi/2])))
        _assert_same(constant.lower, constant.upper)
        p = QuadraticPolynomial(jnp.array([.2, .7, jnp.inf]),
                                jnp.ones((3, 1))*.1, jnp.ones((3, 1, 1))*.02)
        result = pair_trig(kind, PairedQuadratic(p, p, jnp.array([True, False, True])))
        np.testing.assert_array_equal(result.valid, [True, False, False])
        assert bool(jnp.all(jnp.isnan(result.upper.constant[1:])))
        inputs = normalized_pair_seed(jnp.array([-.2, .5]), jnp.array([.2, .9]))
        rule = lambda value: pair_trig(kind, value)
        eager = rule(inputs)
        for actual in (jax.jit(rule)(inputs), jax.jit(jax.vmap(rule))(inputs)):
            for a, b in zip(jax.tree.leaves(actual)[:-1], jax.tree.leaves(eager)[:-1]):
                np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)
            np.testing.assert_array_equal(actual.valid, eager.valid)


def test_pqif_nonsmooth_logic_scatter_and_branch_hull_contain_samples():
    x, y = _vector_inputs()
    native = pqif(_nonsmooth_function)(x, y)
    assert bool(jnp.all(native.valid))
    assert _transform_margin(native, _nonsmooth_function, (x, y)) >= -1e-12
    # An undecided comparison selects through the constant-shift branch hull.
    small_x, small_y = 0.1 * _pair((2, 3), 3, 11), 0.1 * _pair((2, 3), 3, 12)
    hull_function = lambda a, b: jnp.where(a > b, a * a, b - a)
    hull = pqif(hull_function)(small_x, small_y)
    assert bool(jnp.all(hull.valid))
    assert _transform_margin(hull, hull_function, (small_x, small_y)) >= -1e-12


def _control_mixed_function(a, b):
    mask = (a > b).astype(jnp.float64)  # Boolean control -> floating Interval -> constant pair
    nonzero = a.astype(bool)  # pair -> Boolean control [definitely, possibly] nonzero
    scale = jnp.where(a > 0.0, 2, 5).astype(jnp.float64)  # integer cases under an undecided selector
    # A Boolean control mask reaching dot_general directly is relaxed to [0, 1] endpoints.
    masked_dot = lax.dot_general(b, a > b, (([1], [1]), ([], [])), preferred_element_type=jnp.float64)
    return mask * b + jnp.where(nonzero, a, b) + scale * a + masked_dot[:, :1]


def test_control_conversion_selection_and_mask_embedding_contain_samples():
    x, y = 0.1 * _pair((2, 3), 3, 11), 0.1 * _pair((2, 3), 3, 12)
    native = pqif(_control_mixed_function)(x, y)
    assert bool(jnp.all(native.valid))
    assert _transform_margin(native, _control_mixed_function, (x, y)) >= -1e-12


def test_jacfwd_accumulation_and_split_contain_samples():
    x = 0.3 * _pair((3,), 3, 61)
    native = pqif(_derivative_function)(x)
    assert native.shape == (3, 3) and bool(jnp.all(native.valid))
    assert _transform_margin(native, _derivative_function, (x,)) >= -1e-12


def test_rule_jaxpr_size_is_independent_of_source_dimension():
    """No rule unrolls a Python loop over sources: traced size is fixed in m."""
    sizes = []
    for m in (2, 9):
        x, d = _pair((3,), m, 70), _pair((3,), m, 72) + 50.0
        function = lambda a, b: pair_sqrt(pair_square(a * b) / b + 1.0)
        sizes.append(len(jax.make_jaxpr(function)(x, d).jaxpr.eqns))
    assert sizes[0] == sizes[1]


def test_scatter_add_validity_follows_contributing_updates():
    x, _ = _vector_inputs()
    flagged = PairedQuadratic(x.lower, x.upper, jnp.array([True, False, True]))
    function = lambda base, update: base.at[jnp.array([0, 0, 2])].add(update)
    result = pqif(function)(x, flagged)
    np.testing.assert_array_equal(result.valid, [False, True, True])
    overwrite = pqif(lambda base, update: base.at[jnp.array([2])].set(update[1:2]))(x, flagged)
    np.testing.assert_array_equal(overwrite.valid, [True, True, False])


def test_primitive_rules_extend_one_transform_including_nested_jit():
    double_p = jax.extend.core.Primitive("test_pair_double")
    double_p.def_impl(lambda v: 2.0 * v)
    double_p.def_abstract_eval(lambda aval: aval)
    function = lambda v: jax.jit(lambda w: double_p.bind(w) + w)(v)
    x, _ = _vector_inputs()
    with pytest.raises(UnsupportedPairPrimitive, match="test_pair_double"):
        pqif(function)(x)
    result = pqif(function, primitive_rules={double_p: lambda v: 2.0 * v})(x)
    for first, second in zip(jax.tree.leaves(result), jax.tree.leaves(pqif(lambda v: 2.0 * v + v)(x))):
        np.testing.assert_array_equal(first, second)
    assert double_p not in pair_primitive_rules


def test_paired_modules_load_without_taylor():
    directory = Path(__file__).resolve().parents[1] / "immrax" / "inclusion"
    names = ("quadratic_polynomial", "paired_quadratic", "paired_quadratic_transform", "paired_quadratic_selfmap")
    script = (
        "import importlib, sys, types\n"
        "package = types.ModuleType('pair_audit')\n"
        f"package.__path__ = [{str(directory)!r}]\n"
        "sys.modules['pair_audit'] = package\n"
        f"modules = [importlib.import_module('pair_audit.' + name) for name in {names!r}]\n"
        "assert not any(hasattr(module, 'TaylorModel') for module in modules)\n"
        "roots = {'immrax', 'implicit_linearization', 'mujodico', 'mujoco'}\n"
        "forbidden = [n for n in sys.modules if n.split('.')[0] in roots or 'taylor' in n]\n"
        "assert not forbidden, forbidden\n"
        "print(sorted(n for n in sys.modules if n.startswith('pair_audit.')))\n"
    )
    loaded = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert eval(loaded.stdout) == sorted(["pair_audit.interval"] + [f"pair_audit.{name}" for name in names])
