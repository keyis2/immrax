"""Paired-quadratic JAX transform ``pqif``.

``pqif(f)`` traces ``f`` once at point shapes and interprets its Jaxpr
with ``PairedQuadratic`` leaves through an explicit primitive
registry. A primitive without a rule raises
``UnsupportedPairPrimitive``; there is no Taylor fallback.

Validity is per output element: elementwise rules propagate it elementwise
and mixing rules (``reduce_sum``, ``dot_general``, ``scatter-add``) reduce it
over exactly the contributing input axes. Comparisons with a pair operand
return a Boolean ``Interval`` [definitely, possibly]; ``select_n`` selects
exactly where it is decided and uses a constant-shift branch hull
where it is not. Callers extend the registry for one transform through
``pqif(f, primitive_rules=...)``; nested Jaxprs see the same rules.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Literal, primitives

from .interval import Interval
from .paired_quadratic import (
    PairedQuadratic,
    constant_pair,
    pair_abs,
    pair_divide,
    pair_from_interval,
    pair_integer_pow,
    pair_max,
    pair_min,
    pair_select_n,
    pair_sqrt,
    pair_trig,
)
from .quadratic_polynomial import QuadraticPolynomial, quadratic_range


class UnsupportedPairPrimitive(NotImplementedError):
    """A traced primitive reached a paired value without a paired rule."""

    def __init__(self, primitive):
        self.primitive = primitive.name
        super().__init__(f"paired-quadratic primitive unsupported (no fallback): {primitive.name}")


def _is_abstract(value):
    return isinstance(value, (PairedQuadratic, Interval))


def _operand(value):
    """A pair stays a pair; a point becomes an array. Control Intervals stop here."""
    if isinstance(value, Interval):
        raise TypeError("a Boolean control Interval cannot enter paired arithmetic")
    return value if isinstance(value, PairedQuadratic) else jnp.asarray(value)


def _promote(value, like: PairedQuadratic) -> PairedQuadratic:
    value = _operand(value)
    if isinstance(value, PairedQuadratic):
        if value.source_size != like.source_size:
            raise ValueError("paired operands have different source dimensions")
        return value
    return constant_pair(value.astype(like.dtype), like.source_size)


def _template(values) -> PairedQuadratic:
    return next(value for value in values if isinstance(value, PairedQuadratic))


def _map_polynomial(function, x: QuadraticPolynomial) -> QuadraticPolynomial:
    """Apply one value-linear array map to every coefficient slice."""
    over_source = lambda f: jax.vmap(f, in_axes=-1, out_axes=-1)
    return QuadraticPolynomial(function(x.constant), over_source(function)(x.linear),
                               over_source(over_source(function))(x.quadratic))


def _map_pair(function, x: PairedQuadratic, valid) -> PairedQuadratic:
    return PairedQuadratic(_map_polynomial(function, x.lower), _map_polynomial(function, x.upper), valid)


def _structural(primitive, valid_rule=None):
    """Value-linear shape/indexing primitive; trailing operands must be points."""

    def rule(x, *points, **params):
        if any(_is_abstract(point) for point in points):
            raise TypeError(f"paired {primitive.name} requires point-valued indices")
        apply = lambda value: primitive.bind(value, *points, **params)
        if isinstance(x, Interval):  # Boolean control masks are reshaped/indexed too.
            return Interval(apply(x.lower), apply(x.upper))
        valid = valid_rule(x.valid, **params) if valid_rule else apply(x.valid)
        return _map_pair(apply, x, valid)

    return rule


def _reduce_sum_valid(valid, *, axes, **_):
    return jnp.all(valid, axis=axes)


def _convert(x, **params):
    """Conversion: control Intervals map endpointwise; bool gives [definitely, possibly] nonzero."""
    apply = lambda value: lax.convert_element_type_p.bind(value, **params)
    if params["new_dtype"] == jnp.bool_:
        if isinstance(x, Interval):
            low, high, known = x.lower, x.upper, True
        else:
            low, high = quadratic_range(x.lower).lower, quadratic_range(x.upper).upper
            known = x.valid & jnp.isfinite(low) & jnp.isfinite(high)
        definite = (low > 0) | (high < 0)
        possible = ~((low == 0) & (high == 0))
        return Interval(definite & known, possible | ~known)
    if isinstance(x, Interval):
        return Interval(apply(x.lower), apply(x.upper))
    if not jnp.issubdtype(params["new_dtype"], jnp.floating):
        raise TypeError("paired conversion to a non-floating, non-Boolean dtype is unsupported")
    return _map_pair(apply, x, x.valid)


def _add(x, y):
    x, y = _operand(x), _operand(y)
    return x + y if isinstance(x, PairedQuadratic) else y + x


def _sub(x, y):
    x, y = _operand(x), _operand(y)
    return x - y if isinstance(x, PairedQuadratic) else y.__rsub__(x)


def _mul(x, y):
    x, y = _operand(x), _operand(y)
    return x * y if isinstance(x, PairedQuadratic) else y * x


def _div(x, y):
    return pair_divide(_operand(x), _operand(y))


def _split(x, *, sizes, axis):
    """Split coefficients along a value axis; one pair per piece."""
    apply = lambda value: lax.split_p.bind(value, sizes=sizes, axis=axis)
    over_source = lambda f: jax.vmap(f, in_axes=-1, out_axes=-1)

    def pieces(p):
        return zip(apply(p.constant), over_source(apply)(p.linear), over_source(over_source(apply))(p.quadratic))

    return [PairedQuadratic(QuadraticPolynomial(*lower), QuadraticPolynomial(*upper), valid)
            for lower, upper, valid in zip(pieces(x.lower), pieces(x.upper), apply(x.valid))]


def _concatenate(*operands, dimension):
    like = _template(operands)
    values = [_promote(value, like) for value in operands]

    def join(get):
        return jnp.concatenate([get(value) for value in values], axis=dimension)

    def polynomial(end):
        return QuadraticPolynomial(join(lambda v: getattr(v, end).constant),
                                   join(lambda v: getattr(v, end).linear),
                                   join(lambda v: getattr(v, end).quadratic))

    return PairedQuadratic(polynomial("lower"), polynomial("upper"), join(lambda v: v.valid))


def _to_front(x, axes):
    rest = tuple(axis for axis in range(x.ndim) if axis not in axes)
    permutation = tuple(axes) + rest
    return x.transpose(permutation) if isinstance(x, PairedQuadratic) else jnp.transpose(x, permutation)


def _binary_mask_pair(mask: Interval, like: PairedQuadratic) -> PairedQuadratic:
    """Relax a discrete 0/1 control mask to constant endpoints at its dot use.

    For the bilinear summand x_i * m_i the extrema over {0, 1} and [0, 1]
    coincide, so [lower, upper] is a sound local operand where it lies in
    [0, 1]; elsewhere the relaxed element is invalid.
    """
    inside = (mask.lower >= 0) & (mask.upper <= 1) & (mask.lower <= mask.upper)
    numeric = Interval(mask.lower.astype(like.dtype), mask.upper.astype(like.dtype))
    pair = pair_from_interval(numeric, like.source_size)
    return PairedQuadratic(pair.lower, pair.upper, pair.valid & inside)


def _dot_general(a, b, *, dimension_numbers, **_):
    """Broadcast elementwise products, then sum over the contracting axes."""
    like = _template((a, b))
    a, b = (_binary_mask_pair(v, like) if isinstance(v, Interval) else v for v in (a, b))
    a, b = _operand(a), _operand(b)
    (a_contract, b_contract), (a_batch, b_batch) = dimension_numbers
    a = _to_front(a, tuple(a_batch) + tuple(a_contract))
    b = _to_front(b, tuple(b_batch) + tuple(b_contract))
    batch, lead = len(a_batch), len(a_batch) + len(a_contract)
    a_free, b_free = a.shape[lead:], b.shape[lead:]
    a = a.reshape(a.shape + (1,) * len(b_free))
    b = b.reshape(b.shape[:lead] + (1,) * len(a_free) + b_free)
    products = _mul(a, b)
    axes = tuple(range(batch, lead))
    if not axes:
        return products
    return _map_pair(lambda value: jnp.sum(value, axis=axes), products,
                     jnp.all(products.valid, axis=axes))


def _comparison(kind):
    def rule(x, y):
        like = _template((x, y))
        x, y = _promote(x, like), _promote(y, like)
        low = quadratic_range(x.lower - y.upper).lower  # range of x - y
        high = quadratic_range(x.upper - y.lower).upper
        known = x.valid & y.valid & jnp.isfinite(low) & jnp.isfinite(high)
        if kind == "lt":
            definite, possible = high < 0, low < 0
        elif kind == "le":
            definite, possible = high <= 0, low <= 0
        elif kind == "gt":
            definite, possible = low > 0, high > 0
        elif kind == "ge":
            definite, possible = low >= 0, high >= 0
        else:
            equal = (low == 0) & (high == 0)
            maybe_equal = (low <= 0) & (high >= 0)
            definite, possible = (equal, maybe_equal) if kind == "eq" else (~maybe_equal, ~equal)
        return Interval(definite & known, possible | ~known)

    return rule


def _select_n(which, *cases):
    if not any(isinstance(case, PairedQuadratic) for case in cases):
        return _select_control(which, cases)
    if isinstance(which, Interval):
        return pair_select_n(which.lower, which.upper, cases)
    like = _template(cases)
    values = [_promote(case, like) for case in cases]
    shape = values[0].shape
    index = jnp.broadcast_to(jnp.asarray(which), shape)

    def pick(get):
        arrays = [get(value) for value in values]
        extra = arrays[0].ndim - len(shape)
        selector = jnp.broadcast_to(index.reshape(shape + (1,) * extra), arrays[0].shape)
        return lax.select_n(selector, *arrays)

    def polynomial(end):
        return QuadraticPolynomial(pick(lambda v: getattr(v, end).constant),
                                   pick(lambda v: getattr(v, end).linear),
                                   pick(lambda v: getattr(v, end).quadratic))

    valid = lax.select_n(index, *(value.valid for value in values))
    return PairedQuadratic(polynomial("lower"), polynomial("upper"), valid)


def _select_control(which, cases):
    """Interval selection among point/control cases: exact if fixed, else the case hull."""
    intervals = [_control(case) for case in cases]
    which = _control(which)
    shape = jnp.broadcast_shapes(jnp.shape(which.lower), *(jnp.shape(c.lower) for c in intervals))
    lowers = [jnp.broadcast_to(c.lower, shape) for c in intervals]
    uppers = [jnp.broadcast_to(c.upper, shape) for c in intervals]
    index = jnp.broadcast_to(which.lower, shape).astype(jnp.int32)
    fixed = jnp.broadcast_to(which.lower == which.upper, shape)
    return Interval(jnp.where(fixed, lax.select_n(index, *lowers), jnp.min(jnp.stack(lowers), axis=0)),
                    jnp.where(fixed, lax.select_n(index, *uppers), jnp.max(jnp.stack(uppers), axis=0)))


def _control(value):
    return value if isinstance(value, Interval) else Interval(jnp.asarray(value), jnp.asarray(value))


def _boolean_binary(operation):
    def rule(x, y):
        x, y = _control(x), _control(y)
        return Interval(operation(x.lower, y.lower), operation(x.upper, y.upper))
    return rule


def _boolean_not(x):
    return Interval(jnp.logical_not(x.upper), jnp.logical_not(x.lower))


def _boolean_reduce(primitive):
    def rule(x, **params):
        return Interval(primitive.bind(x.lower, **params), primitive.bind(x.upper, **params))
    return rule


def _scatter(primitive):
    """Point-index scatter on coefficients; validity follows the written slots."""

    def rule(operand, indices, updates, **params):
        if _is_abstract(indices):
            raise TypeError(f"paired {primitive.name} requires point-valued indices")
        like = _template((operand, updates))
        operand, updates = _promote(operand, like), _promote(updates, like)
        apply = lambda left, right: primitive.bind(left, indices, right, **params)
        over_source = lambda f: jax.vmap(f, in_axes=(-1, -1), out_axes=-1)

        def polynomial(left, right):
            return QuadraticPolynomial(apply(left.constant, right.constant),
                                       over_source(apply)(left.linear, right.linear),
                                       over_source(over_source(apply))(left.quadratic, right.quadratic))

        if primitive is lax.scatter_p:  # overwrite: each slot takes its writer's validity
            valid = apply(operand.valid, updates.valid)
        else:  # accumulate: a slot is invalid if any contributing update is
            dtype = operand.dtype
            bad = apply(jnp.zeros(operand.shape, dtype), (~updates.valid).astype(dtype))
            valid = operand.valid & (bad == 0)
        return PairedQuadratic(polynomial(operand.lower, updates.lower),
                               polynomial(operand.upper, updates.upper), valid)

    return rule


pair_primitive_rules: dict[Any, Callable[..., Any]] = {
    lax.add_p: _add,
    primitives.add_jaxvals_p: _add,  # JVP tangent accumulation
    lax.sub_p: _sub,
    lax.neg_p: lambda x: -x,
    lax.mul_p: _mul,
    lax.div_p: _div,
    lax.integer_pow_p: lambda x, *, y: pair_integer_pow(x, y),
    lax.convert_element_type_p: _convert,
    lax.copy_p: _structural(lax.copy_p),
    lax.reshape_p: _structural(lax.reshape_p),
    lax.transpose_p: _structural(lax.transpose_p),
    lax.squeeze_p: _structural(lax.squeeze_p),
    lax.broadcast_in_dim_p: _structural(lax.broadcast_in_dim_p),
    lax.slice_p: _structural(lax.slice_p),
    lax.dynamic_slice_p: _structural(lax.dynamic_slice_p),
    lax.gather_p: _structural(lax.gather_p),
    lax.reduce_sum_p: _structural(lax.reduce_sum_p, _reduce_sum_valid),
    lax.concatenate_p: _concatenate,
    lax.split_p: _split,
    lax.dot_general_p: _dot_general,
    lax.lt_p: _comparison("lt"),
    lax.le_p: _comparison("le"),
    lax.gt_p: _comparison("gt"),
    lax.ge_p: _comparison("ge"),
    lax.eq_p: _comparison("eq"),
    lax.ne_p: _comparison("ne"),
    lax.select_n_p: _select_n,
    lax.and_p: _boolean_binary(jnp.logical_and),
    lax.or_p: _boolean_binary(jnp.logical_or),
    lax.not_p: _boolean_not,
    lax.reduce_and_p: _boolean_reduce(lax.reduce_and_p),
    lax.reduce_or_p: _boolean_reduce(lax.reduce_or_p),
    lax.scatter_p: _scatter(lax.scatter_p),
    lax.scatter_add_p: _scatter(lax.scatter_add_p),
    lax.sqrt_p: lambda x, **_: pair_sqrt(x),
    lax.sin_p: lambda x, **_: pair_trig("sin", x),
    lax.cos_p: lambda x, **_: pair_trig("cos", x),
    lax.abs_p: pair_abs,
    lax.max_p: lambda x, y: pair_max(_operand(x), _operand(y)),
    lax.min_p: lambda x, y: pair_min(_operand(x), _operand(y)),
}


def pair_jaxpr(jaxpr, consts, *args, registry=None, source_size=None) -> list[Any]:
    """Evaluate the live output slice; abstract equations use ``registry``.

    Nested ``jit`` calls are interpreted recursively with the same registry.
    With ``source_size``, a floating ``Interval`` produced inside the graph
    (e.g. a converted Boolean mask) enters as a constant endpoint pair, the
    same ingress rule as for transform arguments.
    """
    registry = pair_primitive_rules if registry is None else registry

    def embed(value):
        if (source_size is not None and isinstance(value, Interval)
                and jnp.issubdtype(value.dtype, jnp.floating)):
            return pair_from_interval(value, source_size)
        return value

    live = {v for v in jaxpr.outvars if not isinstance(v, Literal)}
    live_equations = set()
    for index in range(len(jaxpr.eqns) - 1, -1, -1):
        equation = jaxpr.eqns[index]
        if equation.effects or any(v in live for v in equation.outvars):
            live_equations.add(index)
            live.update(v for v in equation.invars if not isinstance(v, Literal))

    env = dict(zip(jaxpr.constvars, consts))
    env.update(zip(jaxpr.invars, args))
    read = lambda v: v.val if isinstance(v, Literal) else env[v]
    for index, equation in enumerate(jaxpr.eqns):
        if index not in live_equations:
            continue
        inputs = [read(v) for v in equation.invars]
        if any(_is_abstract(value) for value in inputs):
            if equation.primitive is primitives.jit_p:
                closed = equation.params["jaxpr"]
                answer = pair_jaxpr(closed.jaxpr, closed.consts, *inputs,
                                                     registry=registry, source_size=source_size)
            elif (rule := registry.get(equation.primitive)) is None:
                raise UnsupportedPairPrimitive(equation.primitive)
            else:
                answer = rule(*inputs, **equation.params)
        else:
            subfuns, params = equation.primitive.get_bind_params(equation.params)
            answer = equation.primitive.bind(*subfuns, *inputs, **params)
        outputs = answer if equation.primitive.multiple_results else (answer,)
        env.update(zip(equation.outvars, map(embed, outputs)))
    return [read(v) for v in jaxpr.outvars]


def pqif(function: Callable[..., Any], *, primitive_rules=None) -> Callable[..., Any]:
    """Transform a point function through ordered quadratic endpoint pairs.

    Positional and keyword ``PairedQuadratic`` leaves must share one source
    dimension. Floating ``Interval`` leaves enter as constant endpoint pairs.
    ``primitive_rules`` maps extra primitives (e.g. caller-owned custom atoms)
    to rules for this transform only; it extends a copy of the core
    registry. Returns one output, or a tuple for several outputs.
    """
    registry = dict(pair_primitive_rules)
    registry.update(primitive_rules or {})

    @wraps(function)
    def wrapped(*args, **kwargs):
        tree = (args, kwargs)
        leaves = jax.tree_util.tree_leaves(tree, is_leaf=_is_abstract)
        if not any(_is_abstract(leaf) for leaf in leaves):
            return function(*args, **kwargs)
        sizes = {leaf.source_size for leaf in leaves if isinstance(leaf, PairedQuadratic)}
        if len(sizes) > 1:
            raise ValueError("paired operands must share one source dimension")
        source_size = sizes.pop() if sizes else 0
        tree = jax.tree_util.tree_map(
            lambda x: pair_from_interval(x, source_size) if isinstance(x, Interval) else x,
            tree, is_leaf=_is_abstract)
        trace = jax.tree_util.tree_map(
            lambda x: x.lower.constant if isinstance(x, PairedQuadratic) else x,
            tree, is_leaf=_is_abstract)
        closed = eqx.filter_make_jaxpr(lambda a, k: function(*a, **k))(*trace)[0]
        flat = [leaf for leaf in jax.tree_util.tree_leaves(tree, is_leaf=_is_abstract)
                if isinstance(leaf, PairedQuadratic) or eqx.is_array(leaf)]
        outputs = pair_jaxpr(closed.jaxpr, closed.consts, *flat, registry=registry,
                                              source_size=source_size)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return wrapped
