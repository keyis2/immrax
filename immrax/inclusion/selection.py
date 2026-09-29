"""Candidate-aware selection of continuous outputs.

``argmin_select`` keeps a discrete ``argmin`` out of the public abstract
domains.  Its ordinary path follows JAX's first-index semantics, while its
inclusion rules compute feasible indices from score bounds and enclose only
the corresponding continuous candidate values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.interpreters import batching

from .affine import AffineBound, constant_affine_bound
from .custom_if import custom_if
from .interval import Interval, interval
from .nif import _argminmax_candidate_mask
from .taylor import (
    TaylorModel,
    constant_taylor_model,
    polynomial_range,
    register_taylor_rule,
    taylor_range,
)


def _validate_shapes(scores, candidates):
    if scores.ndim < 1:
        raise ValueError("argmin_select scores must have a candidate dimension")
    if candidates.ndim < scores.ndim:
        raise ValueError(
            "argmin_select candidates must have shape "
            "batch_shape + (candidate_count,) + selected_shape"
        )
    if candidates.shape[: scores.ndim] != scores.shape:
        raise ValueError(
            "argmin_select scores and candidates must share batch and "
            f"candidate dimensions; got {scores.shape} and {candidates.shape}"
        )


def _take_candidate(values, index, candidate_axis):
    index = jnp.asarray(index)
    index = index.reshape(index.shape + (1,) * (values.ndim - index.ndim))
    index = jnp.broadcast_to(
        index,
        values.shape[:candidate_axis]
        + (1,)
        + values.shape[candidate_axis + 1 :],
    )
    return jnp.squeeze(
        jnp.take_along_axis(values, index, axis=candidate_axis),
        axis=candidate_axis,
    )


def _point_candidate_mask(scores):
    winner = jnp.argmin(scores, axis=-1)
    return jax.nn.one_hot(
        winner, scores.shape[-1], dtype=jnp.bool_
    )


def _score_candidate_mask(scores):
    if isinstance(scores, TaylorModel):
        bounds = taylor_range(scores)
        return _argminmax_candidate_mask(
            bounds.lower, bounds.upper, axis=-1, is_min=True
        )
    if isinstance(scores, AffineBound):
        return _argminmax_candidate_mask(
            scores.lower, scores.upper, axis=-1, is_min=True
        )
    if isinstance(scores, Interval):
        return _argminmax_candidate_mask(
            scores.lower, scores.upper, axis=-1, is_min=True
        )
    return _point_candidate_mask(jnp.asarray(scores))


def _expanded_mask(mask, ndim):
    return mask.reshape(mask.shape + (1,) * (ndim - mask.ndim))


def _common_feasible(values, mask, candidate_axis):
    """Retain coefficients exactly shared by every feasible candidate."""
    reference_index = jnp.argmax(mask.astype(jnp.int32), axis=-1)
    reference = _take_candidate(values, reference_index, candidate_axis)
    value_mask = _expanded_mask(mask, values.ndim)
    common = jnp.all(
        jnp.logical_or(
            jnp.logical_not(value_mask),
            values == jnp.expand_dims(reference, candidate_axis),
        ),
        axis=candidate_axis,
    )
    return jnp.where(common, reference, jnp.zeros_like(reference))


@custom_if
def argmin_select(scores, candidates):
    """Select continuous candidate values using ``argmin(scores)``.

    ``scores`` has shape ``batch_shape + (candidate_count,)`` and
    ``candidates`` has shape
    ``batch_shape + (candidate_count,) + selected_shape``.  Ordinary
    evaluation is exactly JAX ``argmin`` followed by candidate-axis gather.
    Inclusion evaluation returns one model containing all outputs associated
    with feasible winners; the integer index is never approximated by a
    continuous abstract value.
    """

    scores = jnp.asarray(scores)
    candidates = jnp.asarray(candidates)
    _validate_shapes(scores, candidates)
    candidate_axis = scores.ndim - 1
    winner = jnp.argmin(scores, axis=-1)
    return _take_candidate(candidates, winner, candidate_axis)


@argmin_select.defif
def _argmin_select_interval(scores, candidates):
    score_shape = scores.shape
    candidate_shape = candidates.shape
    _validate_shapes(scores, candidates)
    candidate_axis = len(score_shape) - 1
    mask = _expanded_mask(
        _score_candidate_mask(scores), len(candidate_shape)
    )
    candidates = interval(candidates)
    lower = jnp.min(
        jnp.where(mask, candidates.lower, jnp.inf), axis=candidate_axis
    )
    upper = jnp.max(
        jnp.where(mask, candidates.upper, -jnp.inf), axis=candidate_axis
    )
    return Interval(lower, upper)


def _affine_interval(lower, upper, like):
    lower, upper = jnp.broadcast_arrays(lower, upper)
    zeros = jnp.zeros(
        lower.shape + (like.input_size,),
        dtype=jnp.result_type(lower, upper, float),
    )
    return AffineBound(
        zeros,
        lower,
        zeros,
        upper,
        like.domain_lower,
        like.domain_upper,
    )


def _promote_affine(value, like):
    if isinstance(value, AffineBound):
        return value
    if isinstance(value, Interval):
        return _affine_interval(value.lower, value.upper, like)
    return constant_affine_bound(
        value, like.domain_lower, like.domain_upper
    )


def _affine_plane_minimum(coeff, bias, domain_lower, domain_upper):
    return jnp.sum(
        jnp.where(
            coeff >= 0.0,
            coeff * domain_lower,
            coeff * domain_upper,
        ),
        axis=-1,
    ) + bias


def _affine_plane_maximum(coeff, bias, domain_lower, domain_upper):
    return jnp.sum(
        jnp.where(
            coeff >= 0.0,
            coeff * domain_upper,
            coeff * domain_lower,
        ),
        axis=-1,
    ) + bias


@argmin_select.defaif
def _argmin_select_affine(scores, candidates):
    _validate_shapes(scores, candidates)
    like = candidates if isinstance(candidates, AffineBound) else scores
    if not isinstance(like, AffineBound):
        return _argmin_select_interval(scores, candidates)
    candidates = _promote_affine(candidates, like)
    candidate_axis = scores.ndim - 1
    mask = _score_candidate_mask(scores)
    lower_coeff = _common_feasible(
        candidates.lower_coeff, mask, candidate_axis
    )
    upper_coeff = _common_feasible(
        candidates.upper_coeff, mask, candidate_axis
    )

    lower_difference = _affine_plane_minimum(
        candidates.lower_coeff - jnp.expand_dims(lower_coeff, candidate_axis),
        candidates.lower_bias,
        candidates.domain_lower,
        candidates.domain_upper,
    )
    upper_difference = _affine_plane_maximum(
        candidates.upper_coeff - jnp.expand_dims(upper_coeff, candidate_axis),
        candidates.upper_bias,
        candidates.domain_lower,
        candidates.domain_upper,
    )
    value_mask = _expanded_mask(mask, lower_difference.ndim)
    lower_shift = jnp.min(
        jnp.where(value_mask, lower_difference, jnp.inf),
        axis=candidate_axis,
    )
    upper_shift = jnp.max(
        jnp.where(value_mask, upper_difference, -jnp.inf),
        axis=candidate_axis,
    )
    return AffineBound(
        lower_coeff,
        lower_shift,
        upper_coeff,
        upper_shift,
        candidates.domain_lower,
        candidates.domain_upper,
    )


def _promote_taylor(value, like):
    if isinstance(value, TaylorModel):
        return value
    if isinstance(value, Interval):
        shape = value.shape
        zeros = jnp.zeros(
            shape + (like.source_size,),
            dtype=jnp.result_type(value.lower, value.upper, float),
        )
        return TaylorModel(
            jnp.zeros(shape, dtype=zeros.dtype),
            zeros,
            jnp.zeros(shape + (like.source_size, like.source_size), dtype=zeros.dtype),
            value,
        )
    return constant_taylor_model(value, like.source_size)


def _argmin_select_taylor(scores, candidates):
    _validate_shapes(scores, candidates)
    like = candidates if isinstance(candidates, TaylorModel) else scores
    if not isinstance(like, TaylorModel):
        return _argmin_select_interval(scores, candidates)
    candidates = _promote_taylor(candidates, like)
    candidate_axis = scores.ndim - 1
    mask = _score_candidate_mask(scores)
    constant = _common_feasible(
        candidates.constant, mask, candidate_axis
    )
    linear = _common_feasible(
        candidates.linear, mask, candidate_axis
    )
    quadratic = _common_feasible(
        candidates.quadratic, mask, candidate_axis
    )
    difference = TaylorModel(
        candidates.constant - jnp.expand_dims(constant, candidate_axis),
        candidates.linear - jnp.expand_dims(linear, candidate_axis),
        candidates.quadratic - jnp.expand_dims(quadratic, candidate_axis),
        Interval(
            jnp.zeros_like(candidates.constant),
            jnp.zeros_like(candidates.constant),
        ),
    )
    difference_range = polynomial_range(difference)
    lower = difference_range.lower + candidates.remainder.lower
    upper = difference_range.upper + candidates.remainder.upper
    value_mask = _expanded_mask(mask, lower.ndim)
    remainder = Interval(
        jnp.min(jnp.where(value_mask, lower, jnp.inf), axis=candidate_axis),
        jnp.max(jnp.where(value_mask, upper, -jnp.inf), axis=candidate_axis),
    )
    return TaylorModel(constant, linear, quadratic, remainder)


register_taylor_rule(argmin_select.primitive, _argmin_select_taylor)


def _argmin_select_batching_rule(args, batch_axes, **params):
    batch_size = next(
        value.shape[axis]
        for value, axis in zip(args, batch_axes)
        if axis is not batching.not_mapped
    )
    batched = []
    for value, axis in zip(args, batch_axes):
        if axis is batching.not_mapped:
            value = jnp.broadcast_to(value, (batch_size,) + value.shape)
        else:
            value = jnp.moveaxis(value, axis, 0)
        batched.append(value)
    return argmin_select.primitive.bind(*batched, **params), 0


# Keep the atom visible to every inclusion interpreter under ``vmap``.
batching.primitive_batchers[
    argmin_select.primitive
] = _argmin_select_batching_rule


__all__ = ["argmin_select"]
