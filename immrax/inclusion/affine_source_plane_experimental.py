"""Compatibility entry point for the adaptive source-plane product.

The adaptive relaxation is now part of
``AffineBound.product_of_source_planes`` and is used by its production
``"best"`` mode.  This module retains the standalone function used by focused
diagnostics while delegating to the production implementation.
"""

from __future__ import annotations

from .affine import AffineBound

__all__ = ["adaptive_zonotope_source_plane_product"]


def adaptive_zonotope_source_plane_product(
    template: AffineBound,
    first_coeff,
    first_bias,
    second_coeff,
    second_bias,
) -> AffineBound:
    """Bound two exact source planes with the adaptive zonotope rule."""
    return template.product_of_source_planes(
        first_coeff,
        first_bias,
        second_coeff,
        second_bias,
        quadratic_relaxation="adaptive",
    )
