"""Focused tests for finite-candidate selected-output propagation."""

import itertools

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import immrax as irx
from immrax.inclusion import normalized_taylor_seed, taylor_range, tmif


CORE_HALF_SIZE = jnp.asarray([0.4995, 0.4995, 0.4995])
BOX_HALF_SIZE = jnp.asarray([0.5, 0.5, 0.5])
CENTER = jnp.asarray([0.4, 0.4, 0.05])
RADIUS = jnp.asarray([0.01, 0.01, 0.005])


def _candidate_selected_outputs(p):
    face_distance = CORE_HALF_SIZE - jnp.abs(p)
    signs = jnp.where(p >= 0.0, 1.0, -1.0)
    axes = jnp.eye(3, dtype=p.dtype)
    normals = signs[:, None] * axes
    face_positions = (
        p[None, :] * (1.0 - axes)
        + (signs * BOX_HALF_SIZE)[:, None] * axes
    )
    candidates = jnp.concatenate(
        (p[:, None], signs[:, None], normals, face_positions), axis=-1
    )
    return irx.argmin_select(face_distance, candidates)


def _raw_selected_outputs(p):
    face_distance = CORE_HALF_SIZE - jnp.abs(p)
    face_axis = jnp.argmin(face_distance)
    selected = p[face_axis]
    sign = jnp.where(selected >= 0.0, 1.0, -1.0)
    normal = jnp.zeros(3).at[face_axis].set(sign)
    face_position = p.at[face_axis].set(
        sign * BOX_HALF_SIZE[face_axis]
    )
    return jnp.concatenate(
        (jnp.atleast_1d(selected), jnp.atleast_1d(sign), normal, face_position)
    )


def _expected_bounds():
    return (
        np.asarray([0.39, 1.0, 0.0, 0.0, 0.0, 0.39, 0.39, 0.045]),
        np.asarray([0.41, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.055]),
    )


def test_natif_argmin_live_gather_sign_and_replacement_scatter():
    lower, upper = CENTER - RADIUS, CENTER + RADIUS
    box = irx.interval(lower, upper)

    face_axis = irx.natif(
        lambda p: jnp.argmin(CORE_HALF_SIZE - jnp.abs(p))
    )(box)
    selected = irx.natif(_raw_selected_outputs)(box)
    expected_lower, expected_upper = _expected_bounds()

    np.testing.assert_array_equal(face_axis.lower, 0)
    np.testing.assert_array_equal(face_axis.upper, 1)
    np.testing.assert_allclose(selected.lower, expected_lower, atol=1e-7)
    np.testing.assert_allclose(selected.upper, expected_upper, atol=1e-7)
    assert np.all(np.asarray(selected.lower) <= np.asarray(selected.upper))


def test_natif_replacement_scatter_drops_unmatched_discrete_correlation():
    def independently_selected_scatter(index_scores, update_scores):
        destination = jnp.argmin(index_scores)
        update_index = jnp.argmin(update_scores)
        update = lax.dynamic_slice(
            jnp.asarray([10.0, 20.0]), (update_index,), (1,)
        )
        return jnp.zeros(2).at[jnp.expand_dims(destination, 0)].set(update)

    ambiguous = irx.interval(jnp.zeros(2), jnp.ones(2))
    result = irx.natif(independently_selected_scatter)(
        ambiguous, ambiguous
    )

    np.testing.assert_array_equal(result.lower, np.asarray([0.0, 0.0]))
    np.testing.assert_array_equal(result.upper, np.asarray([20.0, 20.0]))


def test_argmin_select_point_tie_jit_and_vmap():
    points = jnp.asarray(
        [
            [0.40, 0.40, 0.05],
            [0.41, 0.41, 0.05],
            [0.39, 0.41, 0.05],
        ]
    )
    expected = jax.vmap(_raw_selected_outputs)(points)

    eager = jax.vmap(_candidate_selected_outputs)(points)
    compiled = jax.jit(jax.vmap(_candidate_selected_outputs))(points)
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(compiled, expected)
    # The equal-distance point must use JAX's first-index winner, axis zero.
    np.testing.assert_array_equal(compiled[1, 2:5], np.asarray([1.0, 0.0, 0.0]))


def test_argmin_select_natural_affine_taylor_and_sample_containment():
    lower, upper = CENTER - RADIUS, CENTER + RADIUS
    box = irx.interval(lower, upper)
    natural = irx.natif(_candidate_selected_outputs)(box)
    affine = irx.affif(
        _candidate_selected_outputs, return_type="affine"
    )(box)
    taylor = tmif(_candidate_selected_outputs)(
        normalized_taylor_seed(lower, upper)
    )
    taylor_bounds = taylor_range(taylor)
    expected_lower, expected_upper = _expected_bounds()

    for bounds in (natural, affine.concretize(), taylor_bounds):
        np.testing.assert_allclose(bounds.lower, expected_lower, atol=1e-7)
        np.testing.assert_allclose(bounds.upper, expected_upper, atol=1e-7)

    source_axis = jnp.linspace(-1.0, 1.0, 9)
    sources = jnp.asarray(list(itertools.product(source_axis, repeat=3)))
    points = CENTER + sources * RADIUS
    samples = jax.jit(jax.vmap(_candidate_selected_outputs))(points)
    for bounds in (natural, affine.concretize(), taylor_bounds):
        assert np.all(np.asarray(samples) >= np.asarray(bounds.lower) - 1e-7)
        assert np.all(np.asarray(samples) <= np.asarray(bounds.upper) + 1e-7)

    # The z face-position component is identical in every feasible branch and
    # therefore keeps its original Taylor source instead of becoming remainder.
    np.testing.assert_allclose(
        taylor.linear[-1], np.asarray([0.0, 0.0, 0.005]), atol=1e-8
    )


def test_argmin_select_inclusion_jit_and_vmap_preserve_atom():
    centers = jnp.asarray([[0.4, 0.4, 0.05], [0.05, 0.4, 0.4]])
    radii = jnp.asarray([[0.01, 0.01, 0.005], [0.005, 0.01, 0.01]])
    lower = (centers - radii).reshape(-1)
    upper = (centers + radii).reshape(-1)

    def batched(flat):
        return jax.vmap(_candidate_selected_outputs)(flat.reshape((2, 3)))

    natural_transform = irx.natif(batched)
    affine_transform = irx.affif(batched, return_type="affine")
    taylor_transform = tmif(batched)

    natural = jax.jit(
        lambda lo, hi: natural_transform(irx.interval(lo, hi))
    )(lower, upper)
    affine = jax.jit(
        lambda lo, hi: affine_transform(irx.interval(lo, hi))
    )(lower, upper)
    taylor = jax.jit(
        lambda lo, hi: taylor_transform(normalized_taylor_seed(lo, hi))
    )(lower, upper)

    for bounds in (natural, affine.concretize(), taylor_range(taylor)):
        assert bounds.shape == (2, 8)
        assert np.all(np.asarray(bounds.lower) <= np.asarray(bounds.upper))
