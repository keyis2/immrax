immrax.inclusion
================

``affif`` returns intervals by default.  Pass ``return_type="affine"`` to keep
an :class:`immrax.inclusion.AffineBound` result for composition with another
``affif`` transform; all affine inputs to a call must share one source domain.

Functions decorated with ``custom_if`` may define ``@f.defif`` for natural
interval inclusion and ``@f.defaif`` for specialized affine inclusion.  When
``defaif`` is absent, affine evaluation transparently expands the original
undecorated implementation without calling ``defif`` or concretizing it.

For expressions already represented as exact affine planes over a shared
source box, :meth:`immrax.inclusion.AffineBound.product_of_source_planes`
provides affine, interval-remainder, and joint-zonotope quadratic relaxations.
Its adaptive mode searches nine zonotope slopes and retains the interval
remainder as a fallback; ``best`` selects complete endpoints between the
adaptive and affine relaxations.
:meth:`immrax.inclusion.AffineBound.product_of_source_plane_and_square`
provides the corresponding centered cubic relaxations for a plane times the
square of another plane.  These methods preserve the source coordinates and
are intended for specialized inclusion rules; their mode strings are static
choices made while JAX traces the caller.

The generic affine multiplication implementation also retains the internal
research switch ``immrax.inclusion.aif._MUL_RELAXATION``.  Its default
``"source_optimized"`` uses the continuous-parameter McCormick rule;
``"baseline"`` restores the original center-selected rule.
Set this switch before tracing or JIT compilation, because changing it does
not invalidate an already compiled JAX executable.

.. automodule:: immrax.inclusion
   :members:
   :undoc-members:
   :show-inheritance:
 
