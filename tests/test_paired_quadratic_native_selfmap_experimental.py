"""Native self-map tube operations against the supported legacy helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from immrax.inclusion import paired_quadratic as legacy
from immrax.inclusion import paired_quadratic_selfmap as legacy_selfmap
from immrax.inclusion.interval import Interval
from immrax.inclusion.paired_quadratic_native_experimental import PairedQuadratic, evaluate_pair
from immrax.inclusion.paired_quadratic_native_selfmap_experimental import (
    append_pair_sources,
    inflate_pair,
    pair_hull,
    pair_inclusion_margins,
    project_pair_sources,
    shared_correction_lift,
    singleton_pair,
)
from immrax.inclusion.quadratic_polynomial_experimental import QuadraticPolynomial, quadratic_range
from immrax.inclusion.taylor import TaylorModel

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


def _legacy(pair):
    zero = jnp.zeros(pair.shape, pair.dtype)
    model = lambda p: TaylorModel(p.constant, p.linear, p.quadratic, Interval(zero, zero))
    return legacy.PairedQuadratic(model(pair.lower), model(pair.upper), pair.valid)


def _same(native, old):
    for end in ("lower", "upper"):
        for field in ("constant", "linear", "quadratic"):
            np.testing.assert_array_equal(getattr(getattr(native, end), field), getattr(getattr(old, end), field))
    np.testing.assert_array_equal(native.valid, old.valid)


FIXTURES = {"affine": lambda: _pair(curvature=0.0), "curved": lambda: _pair(seed=1),
            "curved vector-valued": lambda: _pair((2, 3), 3, seed=2)}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_append_singleton_hull_inflate_and_margins_match_legacy(name):
    first, second = FIXTURES[name](), None
    second = PairedQuadratic(first.lower + 0.05 * _polynomial(first.shape, first.source_size, 9),
                             first.upper + 0.05 * _polynomial(first.shape, first.source_size, 9))
    _same(append_pair_sources(first, 4), legacy_selfmap.append_pair_sources(_legacy(first), 4))
    _same(singleton_pair(first.lower), legacy_selfmap.singleton_pair(_legacy(first).lower))
    _same(pair_hull(first, second), legacy_selfmap.pair_hull(_legacy(first), _legacy(second)))
    _same(inflate_pair(first, 1e-3), legacy_selfmap.inflate_pair(_legacy(first), 1e-3))
    outer = inflate_pair(pair_hull(first, second), 1e-3)
    for mine, theirs in zip(pair_inclusion_margins(outer, first),
                            legacy_selfmap.pair_inclusion_margins(_legacy(outer), _legacy(first))):
        np.testing.assert_array_equal(mine, theirs)
        assert bool(jnp.all(mine > 0))


def test_shared_correction_lift_matches_legacy_and_contains_original():
    value = _pair((3,), 2, seed=3)
    lift, radius = shared_correction_lift(value, 2)
    old_lift, old_radius = legacy_selfmap.shared_correction_lift(_legacy(value), 2)
    _same(lift, old_lift)
    np.testing.assert_array_equal(radius, old_radius)
    assert lift.source_size == 5 and bool(jnp.all(lift.valid))
    # Each output owns one correction source with coefficient rho_i.
    np.testing.assert_array_equal(lift.lower.linear[:, 2:], jnp.diag(radius))
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


def test_projection_ranges_cross_terms_and_matches_legacy():
    value = _pair((3,), 5, seed=5)  # dense quadratic: physical/correction cross terms
    projected, lower_tail, upper_tail = project_pair_sources(value, 2)
    old, old_lower_tail, old_upper_tail = legacy_selfmap.project_pair_sources(_legacy(value), 2)
    _same(projected, old)
    for mine, theirs in ((lower_tail, old_lower_tail), (upper_tail, old_upper_tail)):
        np.testing.assert_array_equal(mine.lower, theirs.lower)
        np.testing.assert_array_equal(mine.upper, theirs.upper)
    assert bool(jnp.all(jnp.abs(value.lower.quadratic[:, :2, 2:]) > 0))  # cross terms were present
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
    batched = jax.vmap(lambda v: inflate_pair(pair_hull(v, project_pair_sources(append_pair_sources(v, 2), 4)[0]),
                                              1e-6))(stack)
    single = _pair((3,), 4, seed=9)
    expected = inflate_pair(pair_hull(single, project_pair_sources(append_pair_sources(single, 2), 4)[0]), 1e-6)
    for first, second in zip(jax.tree.leaves(batched[1]), jax.tree.leaves(expected)):
        np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)


def test_invalid_inputs_follow_per_element_premises():
    value = _pair((3,), 2, seed=11)
    flagged = PairedQuadratic(value.lower, value.upper, jnp.array([True, False, True]))
    lift, radius = shared_correction_lift(flagged, 2)
    np.testing.assert_array_equal(lift.valid, [True, False, True])
    assert not bool(jnp.any(inflate_pair(value, -1.0).valid))
    lower, upper = pair_inclusion_margins(inflate_pair(value, 1e-3), flagged)
    assert bool(jnp.isnan(lower[1]) & jnp.isnan(upper[1])) and bool(jnp.all(jnp.isfinite(lower[::2])))
    np.testing.assert_array_equal(legacy_selfmap.shared_correction_lift(_legacy(flagged), 2)[0].valid,
                                  [True, False, True])
    # A nonfinite coefficient stays local natively; legacy's whole-array
    # finiteness premise invalidates every lifted coordinate.
    nonfinite = PairedQuadratic(QuadraticPolynomial(value.lower.constant.at[1].set(jnp.nan), value.lower.linear,
                                                    value.lower.quadratic), value.upper)
    np.testing.assert_array_equal(shared_correction_lift(nonfinite, 2)[0].valid, [True, False, True])
    assert not bool(jnp.any(legacy_selfmap.shared_correction_lift(_legacy(nonfinite), 2)[0].valid))
