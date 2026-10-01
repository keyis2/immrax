"""Focused analytic fixtures for experimental paired self-map operations."""

import jax
import jax.numpy as jnp
import numpy as np

from immrax.inclusion import (
    Interval, PairedQuadratic, evaluate_pair, pair_status,
)
from immrax.inclusion.paired_quadratic_selfmap_experimental import (
    affine_width_correction_lift, append_pair_sources,
    constant_width_correction_lift, inflate_pair, pair_hull,
    pair_inclusion_margins, project_pair_sources, singleton_pair,
)
from immrax.inclusion.taylor import TaylorModel, evaluate_polynomial

jax.config.update("jax_enable_x64", True)
TOL = 3e-11


def _model(center, linear, quadratic):
    center = jnp.asarray([center], dtype=float)
    linear = jnp.asarray([linear], dtype=float)
    quadratic = jnp.asarray([quadratic], dtype=float)
    return TaylorModel(center, linear, quadratic,
                       Interval(jnp.zeros_like(center), jnp.zeros_like(center)))


def _pair(n=2):
    linear = np.resize(np.array([0.17, -0.11]), n)
    width_linear = np.resize(np.array([0.025, -0.018]), n)
    center = _model(0.05, linear, np.diag(np.resize([0.04, -0.03], n)))
    width = _model(0.23, width_linear,
                   np.diag(np.resize([0.018, -0.012], n)))
    return PairedQuadratic(center - width, center + width)


def _assert_valid(value):
    status = pair_status(value)
    assert all(bool(jnp.all(status[key])) for key in
               ("valid", "zero_remainder", "ordered", "finite")), status


def test_source_append_and_constant_affine_lifts():
    original = _pair()
    _assert_valid(original)
    appended = append_pair_sources(original, 1)
    constant, radius = constant_width_correction_lift(original)
    affine, width_model, quadratic_upper = affine_width_correction_lift(original)
    for value in (appended, constant, affine):
        _assert_valid(value)
        assert value.source_size == 3
    assert bool(jnp.all(quadratic_upper >= 0))
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for xi in sources:
        low, high = evaluate_pair(original, xi)
        midpoint = 0.5 * (low + high)
        half_width = 0.5 * (high - low)
        width = evaluate_polynomial(width_model, xi)
        assert bool(jnp.all(radius >= half_width - TOL))
        assert bool(jnp.all(width >= half_width - TOL))
        for eta in (-1.0, -0.3, 0.0, 0.7, 1.0):
            zeta = jnp.concatenate((xi, jnp.array([eta])))
            np.testing.assert_allclose(evaluate_pair(appended, zeta),
                                       evaluate_pair(original, xi), atol=TOL)
            c_low, c_high = evaluate_pair(constant, zeta)
            a_low, a_high = evaluate_pair(affine, zeta)
            np.testing.assert_allclose(c_low, c_high, atol=TOL)
            np.testing.assert_allclose(a_low, a_high, atol=TOL)
            np.testing.assert_allclose(c_low, midpoint + eta * radius, atol=TOL)
            np.testing.assert_allclose(a_low, midpoint + eta * width, atol=TOL)
    compiled = jax.jit(affine_width_correction_lift)(original)
    for eager, actual in zip(jax.tree.leaves(affine), jax.tree.leaves(compiled[0])):
        np.testing.assert_allclose(eager, actual, atol=TOL)
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), original)
    vmapped = jax.vmap(lambda x: affine_width_correction_lift(x)[0])(batched)
    for eager, actual in zip(jax.tree.leaves(affine),
                             jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(eager, actual, atol=TOL)


def test_projection_margins_hull_and_inflation():
    candidate = _pair()
    full, _, _ = affine_width_correction_lift(candidate)
    full = PairedQuadratic(full.lower - 0.07, full.upper + 0.09)
    projected, lower_tail, upper_tail = project_pair_sources(full, 2)
    _assert_valid(projected)
    assert projected.source_size == 2
    assert bool(jnp.all(lower_tail.lower <= lower_tail.upper))
    assert bool(jnp.all(upper_tail.lower <= upper_tail.upper))
    expanded = inflate_pair(pair_hull(candidate, projected), 1e-9)
    _assert_valid(expanded)
    margins = pair_inclusion_margins(expanded, projected)
    assert bool(jnp.all(margins[0] > 0))
    assert bool(jnp.all(margins[1] > 0))
    sources = jnp.array([[-1., -1.], [-0.4, 0.6], [0., 0.], [1., 1.]])
    for xi in sources:
        projected_low, projected_high = evaluate_pair(projected, xi)
        candidate_low, candidate_high = evaluate_pair(candidate, xi)
        expanded_low, expanded_high = evaluate_pair(expanded, xi)
        assert bool(jnp.all(expanded_low <= jnp.minimum(candidate_low, projected_low) + TOL))
        assert bool(jnp.all(expanded_high >= jnp.maximum(candidate_high, projected_high) - TOL))
        for eta in (-1.0, -0.3, 0.0, 0.7, 1.0):
            zeta = jnp.concatenate((xi, jnp.array([eta])))
            full_low, full_high = evaluate_pair(full, zeta)
            assert bool(jnp.all(projected_low <= full_low + TOL))
            assert bool(jnp.all(projected_high >= full_high - TOL))
    compiled = jax.jit(lambda x: project_pair_sources(x, 2)[0])(full)
    for eager, actual in zip(jax.tree.leaves(projected), jax.tree.leaves(compiled)):
        np.testing.assert_allclose(eager, actual, atol=TOL)
    batched = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), full)
    vmapped = jax.vmap(lambda x: project_pair_sources(x, 2)[0])(batched)
    for eager, actual in zip(jax.tree.leaves(projected),
                             jax.tree.leaves(jax.tree.map(lambda leaf: leaf[0], vmapped))):
        np.testing.assert_allclose(eager, actual, atol=TOL)
    assert singleton_pair(candidate.lower).source_size == 2


def test_selfmap_helpers_have_source_stable_jaxpr_and_visible_invalid_status():
    counts = []
    for n in (2, 8):
        value = _pair(n)
        rules = (
            lambda x: append_pair_sources(x, 1),
            lambda x: constant_width_correction_lift(x)[0],
            lambda x: affine_width_correction_lift(x)[0],
            lambda x: project_pair_sources(append_pair_sources(x, 1), n)[0],
            lambda x: pair_hull(x, inflate_pair(x, 0.02)),
        )
        counts.append([len(jax.make_jaxpr(rule)(value).jaxpr.eqns)
                       for rule in rules])
    assert counts[0] == counts[1]
    bad = PairedQuadratic(_pair().upper, _pair().lower)
    invalid, _ = constant_width_correction_lift(bad)
    assert not bool(jnp.all(pair_status(invalid)["valid"]))
