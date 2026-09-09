immrax.inclusion
================

``affif`` returns intervals by default.  Pass ``return_type="affine"`` to keep
an :class:`immrax.inclusion.AffineBound` result for composition with another
``affif`` transform; all affine inputs to a call must share one source domain.

Functions decorated with ``custom_if`` may define ``@f.defif`` for natural
interval inclusion and ``@f.defaif`` for specialized affine inclusion.  When
``defaif`` is absent, affine evaluation transparently expands the original
undecorated implementation without calling ``defif`` or concretizing it.

.. automodule:: immrax.inclusion
   :members:
   :undoc-members:
   :show-inheritance:
 
