import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir
from jax._src import ad_util

from . import nif


class custom_if:
    """A decorator to define a custom inclusion function for a JAX-traceable function.

    This is analogous to `jax.custom_jvp`.

    This annotation will:
    1. Create a custom primitive that is bound to the implementation of the given function.
    2. Associate the primitive with default batching, lowering, and jvp rules.
    3. Derive the correct shape for abstract evaluation, and associate this shape with the new primitive's abstract eval.
    4. Associate the primitive with a natural interval inclusion function,
       defined using the `@f.defif` decorator.
    5. Associate the primitive with affine inclusion. A specialized rule can
       be defined using `@f.defaif`; otherwise the original implementation is
       expanded through the affine Jaxpr interpreter.
    6. Optionally associate the primitive with a custom JVP rule, defined
       using the `@f.defjvp` decorator.

    For example, to create a primitive for `jnp.polyval` with a custom inclusion function:

    .. code-block:: python

        from immrax.inclusion.custom_if import custom_if
        import jax.numpy as jnp

        @custom_if
        def polyval(a, x):
            return jnp.polyval(a, x)

        @polyval.defif
        def polyval_inclusion(a, x):
            # custom inclusion logic
            ...

        @polyval.defaif
        def polyval_affine_inclusion(a, x):
            # specialized affine inclusion logic (optional)
            ...

        @polyval.defjvp
        def polyval_jvp(primals, tangents):
            # custom derivative logic
            ...

    Now `polyval` can be used in computations: `nif.natif` dispatches to
    `polyval_inclusion`, while `aif.affif` dispatches to the affine rule.  If
    `defaif` is omitted, affine evaluation expands ``fun`` directly and never
    falls back to the ``defif`` interval rule.
    """

    def __init__(self, fun):
        self.fun = fun
        self._if = None
        self._aif = None
        self._jvp = None
        self.primitive = self._create_primitive()
        functools.update_wrapper(self, fun)

    def _create_primitive(self):
        primitive_name = f"{self.fun.__name__}_p"
        primitive = jax.extend.core.Primitive(primitive_name)

        # 1. Implementation
        def impl(*args, **params):
            return self.fun(*args, **params)

        primitive.def_impl(impl)

        # 2. Abstract evaluation (shape inference)
        def default_abstract_eval(*args_aval, **params):
            try:
                shape_args = [
                    jax.ShapeDtypeStruct(arg.shape, arg.dtype) for arg in args_aval
                ]
                shape_dtype = jax.eval_shape(
                    functools.partial(self.fun, **params), *shape_args
                )
                # TODO: I am not entirely sure if this will respect the device / ref counting behavior of the wrapped function
                # Should look here first if those types of problems come up
                return jax.core.ShapedArray(shape_dtype.shape, shape_dtype.dtype)
            except Exception as e:
                raise TypeError(
                    f"Automatic shape inference for '{self.fun.__name__}' failed. "
                ) from e

        primitive.def_abstract_eval(default_abstract_eval)

        # 3. JIT lowering
        # Assuming the wrapped function returns a single result, as in polynomial.py.
        lowering = mlir.lower_fun(self.fun, multiple_results=False)
        mlir.register_lowering(primitive, lowering)

        # 4. Batching rule
        def batching_rule(vector_arg_values, batch_axes, **params):
            res = jax.vmap(
                functools.partial(self.fun, **params),
                in_axes=batch_axes,
            )(*vector_arg_values)

            if isinstance(res, (list, tuple)):
                return res, tuple([0] * len(res))
            else:
                return res, 0

        batching.primitive_batchers[primitive] = batching_rule

        # 5. JVP rules for autodiff
        try:
            sig = inspect.signature(self.fun)
            num_args = sum(
                1
                for param in sig.parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
        except (ValueError, TypeError):
            # Fallback for C functions or other callables without a clear signature
            # This part might need to be adjusted if those functions are used.
            # For now, we won't define JVP rules if we can't inspect the signature.
            num_args = 0

        def jvp_rule(primals, tangents, **params):
            tangents = tuple(ad_util.instantiate(tangent) for tangent in tangents)
            if self._jvp is not None:
                return self._jvp(primals, tangents, **params)
            primal_out, tangent_out = jax.jvp(
                functools.partial(self.fun, **params),
                primals,
                tangents,
            )
            return primal_out, tangent_out

        if num_args:
            ad.primitive_jvps[primitive] = jvp_rule

        # 6. Register inclusion function
        def inclusion_dispatcher(*args, **kwargs):
            if self._if is None:
                raise NotImplementedError(
                    f"No inclusion function defined for '{self.fun.__name__}'. "
                    f"Use '@{self.fun.__name__}.defif' to define it."
                )
            return self._if(*args, **kwargs)

        nif.inclusion_registry[primitive] = inclusion_dispatcher

        # Import here so ``aif`` can finish defining its registry before this
        # module registers custom primitives during package initialization.
        from . import aif

        def affine_inclusion_dispatcher(*args, **params):
            if self._aif is not None:
                return self._aif(*args, **params)

            is_abstract = lambda x: isinstance(x, (aif.AffineBound, nif.Interval))
            trace_args = jax.tree_util.tree_map(
                lambda x: x.lower if is_abstract(x) else x,
                args,
                is_leaf=is_abstract,
            )
            # Trace the undecorated implementation, not ``self``, to avoid
            # recursively binding this same custom primitive.
            closed = eqx.filter_make_jaxpr(
                functools.partial(self.fun, **params)
            )(*trace_args)[0]
            # filter_make_jaxpr closes over non-array Python values. Mirror
            # that filtering here so aif_jaxpr receives exactly one value for
            # each dynamic Jaxpr input variable.
            flat_args = [
                value
                for value in jax.tree_util.tree_leaves(
                    args, is_leaf=is_abstract
                )
                if is_abstract(value) or eqx.is_array(value)
            ]
            outputs = aif.aif_jaxpr(
                closed.jaxpr, closed.literals, *flat_args
            )
            return outputs[0] if len(outputs) == 1 else outputs

        aif.affine_inclusion_registry[primitive] = affine_inclusion_dispatcher

        return primitive

    def __call__(self, *args, **kwargs):
        return self.primitive.bind(*args, **kwargs)

    def defif(self, if_fun):
        """Decorator to define the inclusion function for a custom_if function."""
        self._if = if_fun
        return if_fun

    def defaif(self, aif_fun):
        """Decorator to define a specialized affine inclusion function."""
        self._aif = aif_fun
        return aif_fun

    def defjvp(self, jvp_fun):
        """Decorator to define the JVP rule for a custom_if function."""
        self._jvp = jvp_fun
        return jvp_fun
