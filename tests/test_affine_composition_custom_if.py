import jax
import jax.numpy as jnp
import pytest

import immrax as irx


def test_affif_return_types_and_concretization_agree():
    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0]))
    function = lambda x: 2.0 * x - 1.0

    default_result = irx.affif(function)(box)
    affine_result = irx.affif(function, return_type="affine")(box)

    assert isinstance(default_result, irx.Interval)
    assert isinstance(affine_result, irx.AffineBound)
    assert jnp.allclose(affine_result.concretize().lower, default_result.lower)
    assert jnp.allclose(affine_result.concretize().upper, default_result.upper)


def test_affif_rejects_unknown_return_type():
    with pytest.raises(ValueError, match="return_type"):
        irx.affif(lambda x: x, return_type="linear")


def test_affif_promotes_point_output_in_active_domain():
    box = irx.interval(jnp.array([-1.0, 2.0]), jnp.array([1.0, 3.0]))
    result = irx.affif(
        lambda _x: jnp.array([4.0, -2.0]), return_type="affine"
    )(box)

    assert isinstance(result, irx.AffineBound)
    assert jnp.all(result.lower_coeff == 0)
    assert jnp.all(result.upper_coeff == 0)
    assert jnp.array_equal(result.domain_lower, box.lower)
    assert jnp.array_equal(result.domain_upper, box.upper)


def test_sequential_affif_preserves_sources_and_cancellation():
    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 5.0]))
    first = irx.affif(lambda x: 2.0 * x + 1.0, return_type="affine")(box)
    second = irx.affif(lambda y: 3.0 * y - 4.0, return_type="affine")(first)

    assert jnp.allclose(second.lower_coeff, 6.0 * jnp.eye(2))
    assert jnp.allclose(second.upper_coeff, 6.0 * jnp.eye(2))
    assert jnp.array_equal(second.domain_lower, first.domain_lower)
    assert jnp.array_equal(second.domain_upper, first.domain_upper)

    cancelled = irx.affif(lambda y: y - y, return_type="affine")(first)
    assert jnp.all(cancelled.lower_coeff == 0)
    assert jnp.all(cancelled.upper_coeff == 0)
    assert jnp.all(cancelled.lower == 0)
    assert jnp.all(cancelled.upper == 0)


def test_multiple_affine_inputs_in_one_source_domain():
    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 5.0]))
    source = irx.interval_to_affine_bound(box)
    x = irx.affif(lambda z: 2.0 * z, return_type="affine")(source)
    y = irx.affif(lambda z: -z + 1.0, return_type="affine")(source)

    result = irx.affif(lambda a, b, shift: a + b + shift, return_type="affine")(
        x, y, jnp.array([3.0, 4.0])
    )

    assert jnp.allclose(result.lower_coeff, jnp.eye(2))
    assert jnp.allclose(result.upper_coeff, jnp.eye(2))
    assert jnp.allclose(result.lower_bias, jnp.array([4.0, 5.0]))
    assert jnp.allclose(result.upper_bias, jnp.array([4.0, 5.0]))
    assert jnp.array_equal(result.domain_lower, source.domain_lower)
    assert jnp.array_equal(result.domain_upper, source.domain_upper)


def test_numeric_use_of_three_valued_predicate_is_supported():
    def normalize_pattern(x):
        magnitude = jnp.abs(x)
        return magnitude + 1e-6 * (magnitude == 0.0)

    box = irx.interval(jnp.array([-1.0]), jnp.array([1.0]))
    result = irx.affif(normalize_pattern, return_type="affine")(box)
    concrete = result.concretize()
    samples = jax.vmap(normalize_pattern)(jnp.linspace(-1.0, 1.0, 1001)[:, None])

    assert jnp.all(samples >= concrete.lower)
    assert jnp.all(samples <= concrete.upper)


def test_dot_with_point_branch_selected_by_affine_predicate_is_supported():
    def branch_dot(x):
        direction = jnp.where(
            x[0] >= 0.0,
            jnp.array([1.0, -2.0]),
            jnp.array([-3.0, 4.0]),
        )
        return jnp.dot(x, direction)

    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 1.5]))
    result = irx.affif(branch_dot, return_type="affine")(box)
    concrete = result.concretize()
    samples = jax.vmap(branch_dot)(
        jnp.stack(
            (
                jnp.linspace(-1.0, 2.0, 1001),
                jnp.linspace(0.5, 1.5, 1001),
            ),
            axis=1,
        )
    )

    assert jnp.all(samples >= concrete.lower)
    assert jnp.all(samples <= concrete.upper)


def test_boolean_reduction_of_affine_numeric_array_is_supported():
    def masked_by_any(x):
        return x * jnp.any(x)

    box = irx.interval(jnp.array([-1.0, 0.0]), jnp.array([1.0, 0.0]))
    result = irx.affif(masked_by_any, return_type="affine")(box)
    concrete = result.concretize()
    samples = jax.vmap(masked_by_any)(
        jnp.stack((jnp.linspace(-1.0, 1.0, 1001), jnp.zeros(1001)), axis=1)
    )

    assert jnp.all(samples >= concrete.lower)
    assert jnp.all(samples <= concrete.upper)


def test_affif_rejects_incompatible_and_mixed_source_domains():
    first = irx.interval_to_affine_bound(
        irx.interval(jnp.array([-1.0]), jnp.array([1.0]))
    )
    second = irx.interval_to_affine_bound(
        irx.interval(jnp.array([0.0]), jnp.array([2.0]))
    )
    combine = irx.affif(lambda x, y: x + y, return_type="affine")

    with pytest.raises(ValueError, match="incompatible source domains"):
        combine(first, second)
    with pytest.raises(ValueError, match="cannot currently combine"):
        combine(first, irx.interval(jnp.array([0.0]), jnp.array([1.0])))


def test_custom_if_explicit_defaif_receives_affine_and_point_operands():
    @irx.custom_if
    def atom(x, scale):
        return scale * x + 1.0

    @atom.defif
    def atom_interval(x, scale):
        return scale * x + 1.0

    @atom.defaif
    def atom_affine(x, scale):
        return scale * x + 1.0

    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0]))
    result = irx.affif(lambda x: atom(x, 3.0), return_type="affine")(box)

    assert jnp.allclose(result.lower_coeff, 3.0 * jnp.eye(2))
    assert jnp.allclose(result.upper_coeff, 3.0 * jnp.eye(2))
    assert jnp.allclose(result.lower_bias, jnp.ones(2))
    assert jnp.allclose(result.upper_bias, jnp.ones(2))


def test_custom_if_defaif_can_multiply_two_affine_bounds():
    @irx.custom_if
    def atom(x):
        return jnp.array([[2.0, -1.0]]) @ x

    @atom.defaif
    def atom_affine(x):
        matrix = irx.constant_affine_bound(
            jnp.array([[2.0, -1.0]]), x.domain_lower, x.domain_upper
        )
        return matrix @ x

    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0]))
    result = irx.affif(atom, return_type="affine")(box)

    assert jnp.allclose(result.lower, jnp.array([-8.0]))
    assert jnp.allclose(result.upper, jnp.array([5.0]))


def test_custom_if_affine_fallback_expands_original_and_skips_defif():
    @irx.custom_if
    def atom(x):
        return 3.0 * x + 2.0

    @atom.defif
    def atom_interval(_x):
        raise AssertionError("the natural interval rule must not be used by affif")

    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0]))
    result = irx.affif(atom, return_type="affine")(box)

    assert jnp.allclose(result.lower_coeff, 3.0 * jnp.eye(2))
    assert jnp.allclose(result.upper_coeff, 3.0 * jnp.eye(2))
    assert jnp.allclose(result.lower_bias, 2.0 * jnp.ones(2))
    assert jnp.allclose(result.upper_bias, 2.0 * jnp.ones(2))


def test_custom_if_affine_fallback_filters_static_positional_arguments():
    @irx.custom_if
    def atom(x, scale, unused):
        del unused
        return scale * x + 1.0

    box = irx.interval(jnp.array([-2.0]), jnp.array([3.0]))
    result = irx.affif(
        lambda x: atom(x, 2.0, True), return_type="affine"
    )(box)

    assert jnp.allclose(result.lower_coeff, jnp.array([[2.0]]))
    assert jnp.allclose(result.upper_coeff, jnp.array([[2.0]]))
    assert jnp.allclose(result.lower_bias, jnp.array([1.0]))
    assert jnp.allclose(result.upper_bias, jnp.array([1.0]))


def test_custom_if_jit_vmap_and_jacfwd_affine_dispatch():
    @irx.custom_if
    def slope(x):
        return 2.0 * x

    @irx.custom_if
    def square(x):
        return x * x

    @square.defjvp
    def square_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return x * x, slope(x) * x_dot

    points = jnp.array([-2.0, 0.5, 3.0])
    assert jnp.allclose(jax.jit(slope)(points), 2.0 * points)
    assert jnp.allclose(jax.vmap(slope)(points), 2.0 * points)

    box = irx.interval(jnp.array([-2.0]), jnp.array([3.0]))
    jitted = irx.affif(jax.jit(slope), return_type="affine")(box)
    derivative = irx.affif(jax.jacfwd(square), return_type="affine")(box)

    assert jnp.allclose(jitted.lower_coeff, jnp.array([[2.0]]))
    assert jnp.allclose(jitted.upper_coeff, jnp.array([[2.0]]))
    assert jnp.allclose(derivative.lower_coeff, jnp.array([[2.0]]))
    assert jnp.allclose(derivative.upper_coeff, jnp.array([[2.0]]))


def test_natif_custom_if_behavior_is_unchanged():
    @irx.custom_if
    def atom(x):
        return x

    @atom.defif
    def atom_interval(x):
        return irx.interval(x.lower - 1.0, x.upper + 1.0)

    box = irx.interval(jnp.array([0.0]), jnp.array([2.0]))
    result = irx.natif(atom)(box)

    assert jnp.allclose(result.lower, jnp.array([-1.0]))
    assert jnp.allclose(result.upper, jnp.array([3.0]))
