"""Focused fixtures for the live paired residual operator-closure stages.

Samples compare exact pointwise fibers only as implementation regression.
The inclusion arguments use polynomial ranges and the stated envelopes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from immrax.inclusion import (
    Interval, PairedQuadratic, PairAudit, evaluate_pair, pair_from_interval,
    pair_range, pair_status, pqif,
)
from immrax.inclusion.paired_quadratic import (
    _abs_native, _add, _div_native, _dot_general, _fallback, _max_native,
    _min_native, _mul, _reciprocal_pair, _select_n, _sqrt_native, _trig_native,
    PAIR_NATIVE_FAMILIES,
)
from immrax.inclusion.taylor import TaylorModel

jax.config.update("jax_enable_x64", True)
TOL = 3e-11


def _model(c, a, q):
    c = jnp.asarray(c, dtype=float)
    return TaylorModel(c, jnp.asarray(a, dtype=float), jnp.asarray(q, dtype=float),
                       Interval(jnp.zeros_like(c), jnp.zeros_like(c)))


def _pair(n=2, shift=0.0):
    linear = jnp.resize(jnp.array([0.23, -0.17]), n)
    quadratic = jnp.diag(jnp.resize(jnp.array([0.08, -0.05]), n))
    center = _model(shift, linear, quadratic)
    width = _model(0.22, jnp.resize(jnp.array([0.018, -0.012]), n),
                   jnp.diag(jnp.resize(jnp.array([0.035, 0.025]), n)))
    return PairedQuadratic(center - width, center + width)


def _fibers(value, sources):
    return tuple(np.asarray(x) for x in jax.vmap(
        lambda source: evaluate_pair(value, source))(sources))


def _check(value, exact_lower, exact_upper, sources):
    status = pair_status(value)
    for key in ("valid", "zero_remainder", "ordered", "finite"):
        assert bool(jnp.all(status[key])), (key, status)
    lower, upper = _fibers(value, sources)
    assert np.min(exact_lower - lower) >= -TOL
    assert np.min(upper - exact_upper) >= -TOL
    return float(jnp.max(pair_range(value).width)), float(np.mean(upper - lower))


def _metrics(value, sources):
    lower, upper = _fibers(value, sources)
    return (float(jnp.max(pair_range(value).width)),
            float(np.mean(upper - lower)))


def test_numeric_interval_promotion_and_mixed_arithmetic():
    first = _pair()
    interval = Interval(jnp.array(-0.3), jnp.array(0.45))
    promoted = pair_from_interval(interval, first.source_size)
    assert bool(jnp.all(promoted.lower.linear == 0))
    assert bool(jnp.all(promoted.upper.quadratic == 0))
    assert promoted.lower.constant == interval.lower
    assert promoted.upper.constant == interval.upper
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    xl, xu = _fibers(first, sources)
    for rule, exact in (
        (lambda x: _add(x, interval), (xl - 0.3, xu + 0.45)),
        (lambda x: _mul(x, interval), (
            np.minimum.reduce((xl * -0.3, xl * 0.45, xu * -0.3, xu * 0.45)),
            np.maximum.reduce((xl * -0.3, xl * 0.45, xu * -0.3, xu * 0.45)),
        )),
    ):
        result = rule(first)
        _check(result, *exact, sources)
        compiled = jax.jit(rule)(first)
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(compiled)):
            np.testing.assert_allclose(a, b, atol=TOL)
    native = _mul(first, interval)
    fallback = _fallback(lax.mul_p, first, interval)
    print("mixed product native/Taylor (global, mean fiber) widths:",
          _metrics(native, sources), _metrics(fallback, sources))
    vector = PairedQuadratic(
        _model([0.1, -0.2], [[0.23, -0.17], [-0.12, 0.15]],
               [np.diag([0.08, -0.05]), np.diag([-0.06, 0.04])]),
        _model([0.5, 0.2], [[0.23, -0.17], [-0.12, 0.15]],
               [np.diag([0.08, -0.05]), np.diag([-0.06, 0.04])]),
    )
    coefficients = Interval(jnp.array([0.8, -0.4]), jnp.array([1.0, -0.2]))
    dot = _dot_general(vector, coefficients,
                       dimension_numbers=(((0,), (0,)), ((), ())),
                       precision=None, preferred_element_type=None)
    vector_low, vector_high = _fibers(vector, sources)
    corners = np.stack((vector_low * np.array([0.8, -0.4]),
                        vector_low * np.array([1.0, -0.2]),
                        vector_high * np.array([0.8, -0.4]),
                        vector_high * np.array([1.0, -0.2])))
    _check(dot, corners.min(axis=0).sum(axis=-1),
           corners.max(axis=0).sum(axis=-1), sources)
    dot_fallback = _fallback(lax.dot_general_p, vector, coefficients,
                             dimension_numbers=(((0,), (0,)), ((), ())),
                             precision=None, preferred_element_type=None)
    print("numeric interval dot native/Taylor (global, mean fiber) widths:",
          _metrics(dot, sources), _metrics(dot_fallback, sources))


def test_uncertain_boolean_and_discrete_selection_hulls():
    first, second = _pair(shift=-0.2), _pair(shift=0.35)
    numeric = Interval(jnp.array(-0.08), jnp.array(0.12))
    sources = jnp.array([[-1., -1.], [-0.6, 0.7], [0., 0.], [1., 1.]])
    first_low, first_high = _fibers(first, sources)
    second_low, second_high = _fibers(second, sources)
    uncertain = Interval(jnp.array(False), jnp.array(True))
    native = _select_n(uncertain, first, second)
    native_metrics = _check(native, np.minimum(first_low, second_low),
                            np.maximum(first_high, second_high), sources)
    fallback = _fallback(lax.select_n_p, uncertain, first, second)
    fallback_metrics = _check(fallback, np.minimum(first_low, second_low),
                              np.maximum(first_high, second_high), sources)
    print("uncertain select native/Taylor (global, mean) widths:",
          native_metrics, fallback_metrics)
    fixed = _select_n(Interval(jnp.array(1), jnp.array(1)), first, second)
    for a, b in zip(jax.tree.leaves(fixed.lower), jax.tree.leaves(second.lower)):
        np.testing.assert_allclose(a, b, atol=TOL)
    candidate = _select_n(Interval(jnp.array(0), jnp.array(2)), first,
                          numeric, second)
    _check(candidate, np.minimum.reduce((first_low, np.full(4, -0.08), second_low)),
           np.maximum.reduce((first_high, np.full(4, 0.12), second_high)), sources)
    bad_index = _select_n(Interval(jnp.array(0), jnp.array(3)), first, second)
    assert not bool(jnp.all(pair_status(bad_index)["valid"]))
    bad_branch = _select_n(uncertain, first,
                           PairedQuadratic(second.upper, second.lower))
    assert not bool(jnp.all(pair_status(bad_branch)["valid"]))
    compiled = jax.jit(lambda x, y: _select_n(uncertain, x, y))(first, second)
    for a, b in zip(jax.tree.leaves(native), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(a, b, atol=TOL)
    batched_first = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), first)
    batched_second = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), second)
    vmapped = jax.vmap(lambda x, y: _select_n(uncertain, x, y))(
        batched_first, batched_second)
    for a, b in zip(jax.tree.leaves(native),
                    jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(a, b, atol=TOL)


def test_internal_numeric_interval_provenance_and_selection_closure():
    x = _pair(shift=0.0)
    audit = PairAudit()
    result = pqif(lambda value: jnp.where(value > 0, 1.0, 2.0) + value,
                  operator_strategy="native")(
        x, audit=audit)
    assert isinstance(result, PairedQuadratic)
    assert any(name == "select_n" and kind.startswith("numeric Interval")
               for name, kind, _, _ in audit.interval_events)
    assert any(name == "add" and any(kind.startswith("numeric Interval") for kind in inputs)
               for name, inputs, _ in audit.signatures)
    assert not any(mode == "Taylor fallback" for _, mode, _ in audit.calls)
    sources = jnp.array([[-1., -1.], [0., 0.], [1., 1.]])
    lower, upper = _fibers(x, sources)
    _check(result, lower + 1.0, upper + 2.0, sources)


def test_discrete_binary_mask_in_dot_product():
    vector = PairedQuadratic(
        _model([-0.3, 0.1], [[0.23, -0.17], [-0.12, 0.15]],
               [np.diag([0.08, -0.05]), np.diag([-0.06, 0.04])]),
        _model([0.2, 0.5], [[0.23, -0.17], [-0.12, 0.15]],
               [np.diag([0.08, -0.05]), np.diag([-0.06, 0.04])]),
    )
    mask = Interval(jnp.array([0, 0]), jnp.array([1, 1]))
    params = dict(dimension_numbers=(((0,), (0,)), ((), ())),
                  precision=None, preferred_element_type=None)
    result = _dot_general(vector, mask, **params)
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    low, high = _fibers(vector, sources)
    exact_low = np.minimum(0, low).sum(axis=-1)
    exact_high = np.maximum(0, high).sum(axis=-1)
    _check(result, exact_low, exact_high, sources)
    fallback = _fallback(lax.dot_general_p, vector, mask, **params)
    print("binary mask dot native/Taylor (global, mean fiber) widths:",
          _metrics(result, sources), _metrics(fallback, sources))
    compiled = jax.jit(lambda x: _dot_general(x, mask, **params))(vector)
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(a, b, atol=TOL)
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), vector)
    vmapped = jax.vmap(lambda x: _dot_general(x, mask, **params))(batched)
    for a, b in zip(jax.tree.leaves(result),
                    jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(a, b, atol=TOL)
    bad_mask = Interval(jnp.array([0, 0]), jnp.array([1, 2]))
    bad = _dot_general(vector, bad_mask, **params)
    assert not bool(jnp.all(pair_status(bad)["valid"]))


def test_fixed_sign_reciprocal_and_division():
    x = _pair(shift=0.15)
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    xl, xu = _fibers(x, sources)
    for sign in (1.0, -1.0):
        y = _pair(shift=2.1 * sign)
        yl, yu = _fibers(y, sources)
        reciprocal = _reciprocal_pair(y)
        _check(reciprocal, 1.0 / yu, 1.0 / yl, sources)
        quotient = _div_native(x, y)
        corners = np.stack((xl / yl, xl / yu, xu / yl, xu / yu))
        _check(quotient, corners.min(axis=0), corners.max(axis=0), sources)
        fallback = _fallback(lax.div_p, x, y)
        print("division sign/native/Taylor (global, mean fiber) widths:", sign,
              _metrics(quotient, sources), _metrics(fallback, sources))
        compiled = jax.jit(_div_native)(x, y)
        for a, b in zip(jax.tree.leaves(quotient), jax.tree.leaves(compiled)):
            np.testing.assert_allclose(a, b, atol=TOL)
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
    denominator = _pair(shift=2.1)
    vmapped = jax.vmap(lambda operand: _div_native(operand, denominator))(batched)
    native = _div_native(x, denominator)
    for a, b in zip(jax.tree.leaves(native),
                    jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(a, b, atol=TOL)
    zero_crossing = _pair(shift=0.0)
    invalid = _div_native(x, zero_crossing)
    assert not bool(jnp.all(pair_status(invalid)["valid"]))
    assert bool(jnp.all(jnp.isnan(invalid.lower.constant)))


def test_abs_relu_and_max_min_envelopes():
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for shift in (-2.0, 0.0, 2.0):
        x = _pair(shift=shift)
        low, high = _fibers(x, sources)
        exact_low = np.maximum.reduce((np.zeros_like(low), low, -high))
        exact_high = np.maximum(-low, high)
        result = _abs_native(x)
        _check(result, exact_low, exact_high, sources)
        fallback = _fallback(lax.abs_p, x)
        print("abs shift/native/Taylor (global, mean fiber) widths:", shift,
              _metrics(result, sources), _metrics(fallback, sources))
        compiled = jax.jit(_abs_native)(x)
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(compiled)):
            np.testing.assert_allclose(a, b, atol=TOL)
    x, y = _pair(shift=-0.2), _pair(shift=0.15)
    xl, xu = _fibers(x, sources)
    yl, yu = _fibers(y, sources)
    native_max, native_min = _max_native(x, y), _min_native(x, y)
    _check(native_max, np.maximum(xl, yl), np.maximum(xu, yu), sources)
    _check(native_min, np.minimum(xl, yl), np.minimum(xu, yu), sources)
    print("max native/Taylor (global, mean fiber) widths:",
          _metrics(native_max, sources),
          _metrics(_fallback(lax.max_p, x, y), sources))
    print("min native/Taylor (global, mean fiber) widths:",
          _metrics(native_min, sources),
          _metrics(_fallback(lax.min_p, x, y), sources))
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
    vmapped = jax.vmap(_abs_native)(batched)
    native = _abs_native(x)
    for a, b in zip(jax.tree.leaves(native),
                    jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(a, b, atol=TOL)


def test_sqrt_concave_envelope_and_domain():
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for x in (_pair(shift=1.5),
              PairedQuadratic(_model(0.0, [0.0, 0.0], np.zeros((2, 2))),
                              _model(0.35, [0.02, -0.01],
                                     np.diag([0.01, -0.01])))):
        low, high = _fibers(x, sources)
        result = _sqrt_native(x)
        _check(result, np.sqrt(low), np.sqrt(high), sources)
        fallback = _fallback(lax.sqrt_p, x)
        print("sqrt native/Taylor (global, mean fiber) widths:",
              _metrics(result, sources), _metrics(fallback, sources))
        compiled = jax.jit(_sqrt_native)(x)
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(compiled)):
            np.testing.assert_allclose(a, b, atol=TOL)
    invalid = _sqrt_native(_pair(shift=0.0))
    assert not bool(jnp.all(pair_status(invalid)["valid"]))
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
    vmapped = jax.vmap(_sqrt_native)(batched)
    for a, b in zip(jax.tree.leaves(result),
                    jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(a, b, atol=TOL)


def test_sin_cos_curvature_and_critical_hull():
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for kind in ("sin", "cos"):
        for shift in (0.1, np.pi / 2, np.pi):
            x = _pair(shift=shift)
            low, high = _fibers(x, sources)
            samples = np.linspace(0, 1, 31)[:, None]
            values = (np.sin if kind == "sin" else np.cos)(
                low + samples * (high - low))
            result = _trig_native(kind, x)
            _check(result, values.min(axis=0), values.max(axis=0), sources)
            primitive = lax.sin_p if kind == "sin" else lax.cos_p
            fallback = _fallback(primitive, x)
            print("trig kind/shift/native/Taylor (global, mean fiber) widths:",
                  kind, shift, _metrics(result, sources),
                  _metrics(fallback, sources))
            compiled = jax.jit(lambda operand: _trig_native(kind, operand))(x)
            for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(compiled)):
                np.testing.assert_allclose(a, b, atol=TOL)
        batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), x)
        vmapped = jax.vmap(lambda operand: _trig_native(kind, operand))(batched)
        for a, b in zip(jax.tree.leaves(result),
                        jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
            np.testing.assert_allclose(a, b, atol=TOL)


def test_local_monotone_trig_curvature_is_sound_and_static():
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for kind, shift in (("sin", 0.1), ("cos", 2.0)):
        counts = [len(jax.make_jaxpr(
            lambda x: _trig_native(kind, x, curvature="local")
        )(_pair(n=n, shift=shift)).jaxpr.eqns) for n in (2, 8)]
        assert counts[0] == counts[1]
        operand = _pair(shift=shift)
        low, high = _fibers(operand, sources)
        fibers = low + np.linspace(0, 1, 31)[:, None] * (high - low)
        values = (np.sin if kind == "sin" else np.cos)(fibers)
        local = _trig_native(kind, operand, curvature="local")
        global_rule = _trig_native(kind, operand, curvature="global")
        _check(local, values.min(axis=0), values.max(axis=0), sources)
        assert _metrics(local, sources)[0] <= _metrics(global_rule, sources)[0] + TOL
        compiled = jax.jit(lambda x: _trig_native(kind, x, curvature="local"))(
            operand)
        for first, second in zip(jax.tree.leaves(local), jax.tree.leaves(compiled)):
            np.testing.assert_allclose(first, second, atol=TOL)
        batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), operand)
        vmapped = jax.vmap(lambda x: _trig_native(kind, x, curvature="local"))(
            batched)
        for first, second in zip(jax.tree.leaves(local),
                                 jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
            np.testing.assert_allclose(first, second, atol=TOL)
        interpreted = pqif(
            (lambda x: jnp.sin(x)) if kind == "sin" else (lambda x: jnp.cos(x)),
            operator_strategy="native", trig_curvature="local",
        )(operand)
        for first, second in zip(jax.tree.leaves(local),
                                 jax.tree.leaves(interpreted)):
            np.testing.assert_allclose(first, second, atol=TOL)


def test_native_interpreter_operator_chain_has_no_taylor_fallback():
    x, y = _pair(shift=0.0), _pair(shift=2.2)

    @jax.jit
    def chain(a, b):
        selected = jnp.where(a > 0, a, -a)
        numeric = jnp.where(a > 0, 1.0, 2.0)
        return (jnp.sin(a) + jnp.cos(a) + jnp.abs(a)
                + jnp.sqrt(b) + selected + a / b + numeric * a)

    audit = PairAudit()
    result = pqif(chain, operator_strategy="native")(x, y, audit=audit)
    status = pair_status(result)
    assert all(bool(jnp.all(status[key])) for key in
               ("valid", "zero_remainder", "ordered", "finite"))
    assert not any(mode == "Taylor fallback" for _, mode, _ in audit.calls)
    assert {"sin", "cos", "abs", "sqrt", "div", "select_n", "mul"}.issubset(
        {name for name, _, _ in audit.signatures})
    legacy_audit = PairAudit()
    pqif(chain, operator_strategy="legacy")(x, y, audit=legacy_audit)
    print("chain native/legacy fallback events:",
          sum(mode == "Taylor fallback" for _, mode, _ in audit.calls),
          sum(mode == "Taylor fallback" for _, mode, _ in legacy_audit.calls))
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for source in sources:
        xl, xu = evaluate_pair(x, source)
        yl, yu = evaluate_pair(y, source)
        lower, upper = evaluate_pair(result, source)
        for fraction in (0.0, 0.4, 1.0):
            xv = xl + fraction * (xu - xl)
            yv = yl + (1.0 - fraction) * (yu - yl)
            exact = chain(xv, yv)
            assert bool(lower <= exact + TOL)
            assert bool(exact <= upper + TOL)


def test_native_scalar_rule_jaxpr_size_is_source_dimension_stable():
    rules = (
        lambda x, y: _div_native(x, y),
        lambda x, y: _abs_native(x),
        lambda x, y: _sqrt_native(y),
        lambda x, y: _trig_native("sin", x),
        lambda x, y: _select_n(Interval(jnp.array(False), jnp.array(True)), x, y),
    )
    counts = []
    for size in (2, 8):
        x, y = _pair(n=size, shift=0.0), _pair(n=size, shift=2.2)
        current = []
        for rule in rules:
            result = rule(x, y)
            status = pair_status(result)
            assert all(bool(jnp.all(status[key])) for key in
                       ("valid", "zero_remainder", "ordered", "finite"))
            current.append(len(jax.make_jaxpr(rule)(x, y).jaxpr.eqns))
        counts.append(current)
    print("native scalar rule Jaxpr equations n=2/n=8:", counts)
    assert counts[0] == counts[1]


def test_static_native_family_selector_preserves_legacy_and_native_endpoints():
    x, y = _pair(shift=0.0), _pair(shift=2.2)

    @jax.jit
    def chain(a, b):
        return (jnp.sin(a) + jnp.abs(a) + jnp.sqrt(b) + a / b
                + jnp.where(a > 0, a, -a))

    legacy = pqif(chain, operator_strategy="legacy")(x, y)
    empty = pqif(chain, operator_strategy="native", native_families=())(x, y)
    native = pqif(chain, operator_strategy="native")(x, y)
    all_families = pqif(chain, operator_strategy="legacy",
                        native_families=PAIR_NATIVE_FAMILIES)(x, y)
    for expected, actual in ((legacy, empty), (native, all_families)):
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_allclose(a, b, atol=TOL)
    audit = PairAudit(capture_pairs=True)
    partial = pqif(chain, operator_strategy="legacy",
                   native_families=("trigonometry",))(x, y, audit=audit)
    assert bool(jnp.all(pair_status(partial)["valid"]))
    assert any(name == "sin" for name, _ in audit.pair_outputs)
    assert any(name == "abs" and mode == "Taylor fallback"
               for name, mode, _ in audit.calls)
    assert not any(name == "sin" and mode == "Taylor fallback"
                   for name, mode, _ in audit.calls)
