"""Paired-quadratic self-map source and tube operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion.paired_quadratic import PairedQuadratic, evaluate_pair
from immrax.inclusion.paired_quadratic_selfmap import (
    append_pair_sources,
    inflate_pair,
    pair_hull,
    pair_inclusion_margins,
    project_pair_sources,
    shared_correction_lift,
    singleton_pair,
)
from immrax.inclusion.quadratic_polynomial import QuadraticPolynomial, quadratic_range

jax.config.update("jax_enable_x64", True)


def _polynomial(shape, m, seed, curvature=1.0):
    rng = np.random.default_rng(seed)
    return QuadraticPolynomial(rng.normal(size=shape), rng.normal(size=shape + (m,)),
                               curvature * rng.normal(size=shape + (m, m)))


def _pair(shape=(3,), m=2, seed=0, curvature=1.0):
    """Ordered pair C -/+ W, W a positive constant plus a PSD quadratic form."""
    center = _polynomial(shape, m, seed, curvature)
    rng = np.random.default_rng(seed + 50)
    factor = rng.normal(size=shape + (m, m))
    width = QuadraticPolynomial(rng.uniform(0.05, 0.2, size=shape), np.zeros(shape + (m,)),
                                curvature * np.einsum("...ki,...kj->...ij", factor, factor))
    return PairedQuadratic(center - width, center + width)


def _contains(outer, inner, m, samples=64, seed=1):
    """Sampled check that the inner endpoint functions lie inside the outer ones."""
    for point in np.random.default_rng(seed).uniform(-1, 1, (samples, m)):
        outer_low, outer_high = evaluate_pair(outer, point)
        inner_low, inner_high = evaluate_pair(inner, point)
        assert bool(jnp.all(outer_low <= inner_low + 1e-12) & jnp.all(inner_high <= outer_high + 1e-12))


FIXTURES = {"affine": lambda: _pair(curvature=0.0), "curved": lambda: _pair(seed=1),
            "curved vector-valued": lambda: _pair((2, 3), 3, seed=2)}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_append_singleton_hull_inflate_and_margins(name):
    first = FIXTURES[name]()
    shift = 0.05 * _polynomial(first.shape, first.source_size, 9)
    second = PairedQuadratic(first.lower + shift, first.upper + shift)
    appended = append_pair_sources(first, 4)
    assert appended.source_size == first.source_size + 4
    np.testing.assert_array_equal(appended.lower.linear[..., first.source_size:], 0.0)
    np.testing.assert_array_equal(appended.upper.quadratic[..., first.source_size:, :], 0.0)
    singleton = singleton_pair(first.lower)
    assert singleton.lower is singleton.upper and bool(jnp.all(singleton.valid))
    hull = pair_hull(first, second)
    _contains(hull, first, first.source_size)
    _contains(hull, second, first.source_size)
    inflated = inflate_pair(first, 1e-3)
    np.testing.assert_array_equal(inflated.lower.constant, first.lower.constant - 1e-3)
    np.testing.assert_array_equal(inflated.upper.linear, first.upper.linear)
    lower, upper = pair_inclusion_margins(inflate_pair(hull, 1e-3), first)
    assert bool(jnp.all(lower > 0) & jnp.all(upper > 0))


def test_shared_correction_lift_radius_contract_and_containment():
    value = _pair((3,), 2, seed=3)
    lift, radius = shared_correction_lift(value, 2, outward_ulp_steps=0)
    midpoint = 0.5 * (value.lower + value.upper)
    expected = jnp.maximum(quadratic_range(midpoint - value.lower).upper, quadratic_range(value.upper - midpoint).upper)
    np.testing.assert_array_equal(radius, expected)
    # The supported lift adds two outward ULPs; projection rounding needs them.
    lift, stepped = shared_correction_lift(value, 2)
    np.testing.assert_array_equal(stepped, jnp.nextafter(jnp.nextafter(radius, jnp.inf), jnp.inf))
    assert lift.source_size == 5 and bool(jnp.all(lift.valid)) and lift.lower is lift.upper
    # Each output owns one correction source with coefficient rho_i.
    np.testing.assert_array_equal(lift.lower.linear[:, 2:], jnp.diag(stepped))
    projected, _, _ = project_pair_sources(lift, 2)
    assert all(bool(jnp.all(margin >= 0)) for margin in pair_inclusion_margins(projected, value))


def test_lift_radius_covers_midpoint_rounding_where_half_width_fails():
    """Curved pair whose constants differ by one ULP: fl((L+U)/2) rounds to L.

    Then sup(U - m) = ulp, while the older sup((U - L)/2) = ulp/2 undercounts.
    """
    base = _polynomial((2,), 2, seed=4)
    constant = jnp.array([1.0, 3.0])
    lower = QuadraticPolynomial(constant, base.linear, base.quadratic)
    upper = QuadraticPolynomial(jnp.nextafter(constant, jnp.inf), base.linear, base.quadratic)
    value = PairedQuadratic(lower, upper)
    lift, radius = shared_correction_lift(value, 2, outward_ulp_steps=0)
    projected, _, _ = project_pair_sources(lift, 2)
    assert all(bool(jnp.all(margin >= 0)) for margin in pair_inclusion_margins(projected, value))
    old_radius = quadratic_range(0.5 * (value.upper - value.lower)).upper
    assert bool(jnp.all(old_radius < radius))
    midpoint = 0.5 * (value.lower + value.upper)
    old_lift = PairedQuadratic(*(QuadraticPolynomial(
        midpoint.constant, jnp.pad(midpoint.linear, ((0, 0), (0, 2))).at[jnp.arange(2), 2 + jnp.arange(2)].add(old_radius),
        jnp.pad(midpoint.quadratic, ((0, 0), (0, 2), (0, 2)))),) * 2)
    old_projected, _, _ = project_pair_sources(old_lift, 2)
    assert bool(jnp.any(pair_inclusion_margins(old_projected, value)[1] < 0))


def test_projection_ranges_every_eliminated_term_including_cross_terms():
    value = _pair((3,), 5, seed=5)  # dense quadratic: physical/correction cross terms
    projected, lower_tail, upper_tail = project_pair_sources(value, 2)
    assert bool(jnp.all(jnp.abs(value.lower.quadratic[:, :2, 2:]) > 0))  # cross terms were present
    np.testing.assert_array_equal(projected.lower.linear, value.lower.linear[:, :2])
    np.testing.assert_array_equal(projected.upper.quadratic, value.upper.quadratic[:, :2, :2])
    np.testing.assert_array_equal(projected.lower.constant, value.lower.constant + lower_tail.lower)
    np.testing.assert_array_equal(projected.upper.constant, value.upper.constant + upper_tail.upper)
    rng = np.random.default_rng(6)
    for _ in range(64):  # projection contains every slice over the eliminated sources
        point = rng.uniform(-1, 1, 5)
        low, high = evaluate_pair(value, point)
        p_low, p_high = evaluate_pair(projected, point[:2])
        assert bool(jnp.all(p_low <= low + 1e-12) & jnp.all(high <= p_high + 1e-12))


def test_helpers_under_jit_and_outer_vmap():
    value = _pair((3,), 2, seed=7)
    eager = shared_correction_lift(value, 2)
    jitted = jax.jit(lambda v: shared_correction_lift(v, 2))(value)
    for first, second in zip(jax.tree.leaves(jitted), jax.tree.leaves(eager)):
        np.testing.assert_array_equal(first, second)
    stack = jax.tree.map(lambda *leaves: jnp.stack(leaves), *(_pair((3,), 4, seed=s) for s in (8, 9, 10)))
    update = lambda v: inflate_pair(pair_hull(v, project_pair_sources(append_pair_sources(v, 2), 4)[0]), 1e-6)
    batched = jax.jit(jax.vmap(update))(stack)
    expected = update(_pair((3,), 4, seed=9))
    for first, second in zip(jax.tree.leaves(batched[1]), jax.tree.leaves(expected)):
        np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)


def test_invalid_inputs_follow_per_element_premises():
    value = _pair((3,), 2, seed=11)
    flagged = PairedQuadratic(value.lower, value.upper, jnp.array([True, False, True]))
    lift, _ = shared_correction_lift(flagged, 2)
    np.testing.assert_array_equal(lift.valid, [True, False, True])
    nonfinite = PairedQuadratic(QuadraticPolynomial(value.lower.constant.at[1].set(jnp.nan), value.lower.linear,
                                                    value.lower.quadratic), value.upper)
    np.testing.assert_array_equal(shared_correction_lift(nonfinite, 2)[0].valid, [True, False, True])
    assert not bool(jnp.any(inflate_pair(value, -1.0).valid))
    lower, upper = pair_inclusion_margins(inflate_pair(value, 1e-3), flagged)
    assert bool(jnp.isnan(lower[1]) & jnp.isnan(upper[1])) and bool(jnp.all(jnp.isfinite(lower[::2])))
    with pytest.raises(ValueError):
        shared_correction_lift(value, 3)
