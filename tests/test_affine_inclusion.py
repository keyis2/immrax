import importlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lax import linalg as LA

import immrax as irx

aif_module = importlib.import_module("immrax.inclusion.aif")


def _check_samples(function, box, enclosure, samples=41):
    axes = [
        jnp.linspace(l, u, samples)
        for l, u in zip(box.lower.reshape(-1), box.upper.reshape(-1))
    ]
    if box.size == 1:
        points = axes[0][:, None].reshape((samples,) + box.shape)
    else:
        # A modest deterministic grid catches dependency and broadcasting bugs
        # without making the test suite expensive.
        reduced = [axis[jnp.linspace(0, samples - 1, 7).astype(int)] for axis in axes]
        points = jnp.asarray(list(itertools.product(*reduced))).reshape(
            (-1,) + box.shape
        )
    values = jax.vmap(function)(points)
    assert jnp.all(values >= enclosure.lower - 1e-5)
    assert jnp.all(values <= enclosure.upper + 1e-5)


def test_affine_bound_lift_and_concretization():
    box = irx.interval(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0]))
    bound = irx.interval_to_affine_bound(box)
    assert isinstance(bound, irx.AffineBound)
    assert bound.lower_coeff.shape == (2, 2)
    assert jnp.allclose(bound.lower_coeff, jnp.eye(2))
    assert jnp.allclose(bound.lower, box.lower)
    assert jnp.allclose(bound.upper, box.upper)
    assert jnp.allclose(irx.affine_bound_to_interval(bound).lower, box.lower)


def test_affif_preserves_linear_dependency_and_interval_api():
    box = irx.interval(jnp.array([-1.0, 2.0]), jnp.array([3.0, 5.0]))
    result = irx.affif(lambda x: x - x)(box)
    natural = irx.natif(lambda x: x - x)(box)
    assert isinstance(result, irx.Interval)
    assert jnp.all(result.lower == 0)
    assert jnp.all(result.upper == 0)
    assert jnp.all(natural.width > result.width)


def test_affif_multiple_interval_arguments_share_one_domain():
    x = irx.interval(jnp.array([-1.0, 2.0]), jnp.array([2.0, 4.0]))
    u = irx.interval(jnp.array([-3.0]), jnp.array([1.0]))
    function = lambda x, u: jnp.array([[1.0, -2.0]]) @ x + 3.0 * u
    result = irx.affif(function)(x, u)
    assert jnp.allclose(result.lower, jnp.array([-18.0]))
    assert jnp.allclose(result.upper, jnp.array([1.0]))


def test_affif_fixed_positional_and_constant_keyword_arguments():
    box = irx.interval(jnp.array([1.0]), jnp.array([2.0]))
    function = lambda x, scale, shift=0.0: scale * x + shift
    result = irx.affif(function, fixed_argnums=1)(box, jnp.array([3.0]), shift=2.0)
    assert jnp.allclose(result.lower, jnp.array([5.0]))
    assert jnp.allclose(result.upper, jnp.array([8.0]))


@pytest.mark.parametrize(
    ("function", "lower", "upper"),
    [
        (lambda x: jnp.exp(x), -1.0, 1.5),
        (lambda x: jnp.log(x), 0.2, 3.0),
        (lambda x: jnp.log1p(x), -0.8, 2.0),
        (lambda x: jnp.sqrt(x), 0.0, 4.0),
        (lambda x: 1 / x, 0.2, 3.0),
        (lambda x: x**4, -2.0, 1.0),
        (lambda x: jnp.abs(x), -2.0, 3.0),
        (lambda x: jnp.tanh(x), -2.0, 2.0),
        (lambda x: jnp.sin(x), -2.0, 2.0),
        (lambda x: jnp.cos(x), -2.0, 2.0),
        (lambda x: jnp.tan(x), -0.7, 0.9),
        (lambda x: jnp.arctan(x), -3.0, 2.0),
        (lambda x: jnp.arcsin(x), -0.9, 0.8),
        (lambda x: x**1.5, 0.1, 3.0),
    ],
)
def test_affif_unary_relaxations_are_sound(function, lower, upper):
    box = irx.interval(jnp.array([lower]), jnp.array([upper]))
    result = irx.affif(function)(box)
    assert isinstance(result, irx.Interval)
    assert jnp.all(result.lower <= result.upper)
    _check_samples(function, box, result)


@pytest.mark.parametrize(
    ("function", "values"),
    [
        (jnp.exp, [-2.0, 0.0, 1.5]),
        (jnp.log, [0.2, 1.0, 3.0]),
        (jnp.log1p, [-0.8, 0.0, 2.0]),
        (jnp.sqrt, [0.0, 0.25, 1.0, 4.0]),
        (lambda x: 1 / x, [-3.0, -0.2, 0.2, 3.0]),
        (lambda x: x**4, [-2.0, 0.0, 3.0]),
        (jnp.abs, [-2.0, 0.0, 3.0]),
        (jnp.tanh, [-2.0, 0.0, 2.0]),
        (jnp.sin, [-2.0, 0.0, 2.0]),
        (jnp.cos, [-2.0, 0.0, 2.0]),
        (jnp.tan, [-0.9, 0.0, 0.9]),
        (jnp.arctan, [-3.0, 0.0, 2.0]),
        (jnp.arcsin, [-1.0, -0.9, 0.0, 0.8, 1.0]),
        (lambda x: x**1.5, [0.1, 1.0, 3.0]),
    ],
)
def test_affif_unary_point_inputs_are_exact(function, values):
    point = jnp.asarray(values)
    result = irx.affif(function)(irx.interval(point, point))
    expected = function(point)

    assert jnp.allclose(result.lower, expected)
    assert jnp.allclose(result.upper, expected)


@pytest.mark.parametrize(
    ("function", "left", "right"),
    [
        (jnp.power, 0.0, 2.0),
        (jnp.power, 2.0, 0.5),
        (jnp.arctan2, 0.0, 0.0),
        (jnp.arctan2, 1.0, -1.0),
    ],
)
def test_affif_binary_point_inputs_are_exact(function, left, right):
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    result = irx.affif(function)(
        irx.interval(left, left), irx.interval(right, right)
    )
    expected = function(left, right)

    assert jnp.allclose(result.lower, expected)
    assert jnp.allclose(result.upper, expected)


def test_affif_point_valued_intermediate_nonlinear_ops_are_exact():
    box = irx.interval(jnp.array([-1.0]), jnp.array([1.0]))

    def function(source):
        zero = 0.0 * source[0]
        one = zero + 1.0
        return jnp.array(
            [
                jnp.sqrt(one),
                jnp.tan(zero + 0.9),
                jnp.arcsin(one),
                jnp.power(zero, zero + 2.0),
                jnp.arctan2(zero, zero),
            ]
        )

    result = irx.affif(function)(box)
    expected = function(jnp.zeros_like(box.lower))

    assert jnp.allclose(result.lower, expected)
    assert jnp.allclose(result.upper, expected)


def test_affif_tensor_operations_and_uncertain_dot_product():
    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))

    def function(x):
        y = jnp.concatenate((x, x[::-1]))
        return jnp.array([1.0, -2.0, 0.5, 3.0]) @ y + x @ x

    result = irx.affif(function)(box)
    assert result.shape == ()
    _check_samples(function, box, result)


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((2,), (2,)),
        ((2, 2), (2,)),
        ((2,), (2, 2)),
        ((2, 2), (2, 2)),
    ],
)
def test_affine_bound_matmul_operator_uses_dot_general_rule(
    left_shape, right_shape
):
    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))
    source = irx.interval_to_affine_bound(box)
    left_size = int(np.prod(left_shape))
    right_size = int(np.prod(right_shape))
    left_indices = jnp.arange(left_size) % source.size
    right_indices = jnp.arange(right_size) % source.size
    left = source[left_indices].reshape(left_shape)
    right = source[::-1][right_indices].reshape(right_shape)

    direct = left @ right
    transformed = irx.affif(
        jnp.matmul, return_type="affine"
    )(left, right)

    assert isinstance(direct, irx.AffineBound)
    assert jnp.allclose(direct.lower, transformed.lower)
    assert jnp.allclose(direct.upper, transformed.upper)


def test_affif_uncertain_index_and_multiway_selection():
    values = irx.interval(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 4.0, 6.0]))
    index = irx.interval(jnp.array(0, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32))
    gathered = irx.affif(lambda x, i: x[i])(values, index)
    assert gathered.lower <= 1.0
    assert gathered.upper >= 6.0

    branches = (lambda x: x, lambda x: -x, lambda x: x**2)
    selected = irx.affif(lambda x, i: jax.lax.switch(i, branches, x))(values, index)
    for i in range(3):
        for endpoint in (values.lower, values.upper):
            concrete = branches[i](endpoint)
            assert jnp.all(concrete >= selected.lower)
            assert jnp.all(concrete <= selected.upper)


def test_affif_reductions_and_selection_are_sound():
    box = irx.interval(jnp.array([-2.0, -0.5, 1.0]), jnp.array([1.0, 2.0, 3.0]))

    def function(x):
        selected = jnp.where(x >= 0, x, -x)
        return jnp.array([jnp.sum(selected), jnp.max(x), jnp.min(x)])

    result = irx.affif(function)(box)
    _check_samples(function, box, result)


def test_affif_finitely_bounded_while_loop():
    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))

    def function(x):
        def condition(state):
            iteration, _ = state
            return iteration < 3

        def body(state):
            iteration, value = state
            return iteration + 1, value + x

        return jax.lax.while_loop(condition, body, (0, jnp.zeros_like(x)))[1]

    result = irx.affif(function)(box)
    assert jnp.allclose(result.lower, 3 * box.lower)
    assert jnp.allclose(result.upper, 3 * box.upper)


def test_affif_cond_and_finite_scan_are_sound():
    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))

    def conditional(x):
        return jax.lax.cond(x[0] > 0, lambda z: z * z, lambda z: -z, x)

    conditional_result = irx.affif(conditional)(box)
    _check_samples(conditional, box, conditional_result)

    def recurrence(x):
        def step(carry, multiplier):
            carry = carry + multiplier * x
            return carry, carry

        return jax.lax.scan(step, jnp.zeros_like(x), jnp.arange(3.0))[1]

    recurrence_result = irx.affif(recurrence)(box)
    _check_samples(recurrence, box, recurrence_result)


def test_affif_domain_failures_return_top():
    log_result = irx.affif(jnp.log)(irx.interval(jnp.array([-1.0]), jnp.array([2.0])))
    reciprocal_result = irx.affif(lambda x: 1 / x)(
        irx.interval(jnp.array([-1.0]), jnp.array([2.0]))
    )
    assert jnp.isneginf(log_result.lower[0]) and jnp.isposinf(log_result.upper[0])
    assert jnp.isneginf(reciprocal_result.lower[0]) and jnp.isposinf(
        reciprocal_result.upper[0]
    )


def test_affif_cholesky_constant_is_exact_and_uncertain_family_is_sound():
    constant_matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    constant_result = irx.affif(jnp.linalg.cholesky)(irx.interval(constant_matrix))
    expected = jnp.linalg.cholesky(constant_matrix)
    assert jnp.allclose(constant_result.lower, expected)
    assert jnp.allclose(constant_result.upper, expected)

    matrix = irx.interval(
        jnp.array([[3.0, -0.2], [-0.2, 2.0]]),
        jnp.array([[4.0, 0.2], [0.2, 3.0]]),
    )
    result = irx.affif(jnp.linalg.cholesky)(matrix)
    corners = jnp.asarray(
        list(
            itertools.product(*zip(matrix.lower.reshape(-1), matrix.upper.reshape(-1)))
        )
    ).reshape((-1, 2, 2))
    values = jax.vmap(jnp.linalg.cholesky)(corners)
    assert jnp.all(values >= result.lower - 1e-5)
    assert jnp.all(values <= result.upper + 1e-5)
    assert jnp.all(result.lower <= result.upper)


def test_affif_cholesky_domain_failure_returns_top():
    matrix = irx.interval(
        jnp.array([[-1.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, 2.0]]),
    )
    result = irx.affif(jnp.linalg.cholesky)(matrix)
    assert jnp.all(jnp.isneginf(jnp.diag(result.lower)))
    assert jnp.all(jnp.isposinf(jnp.diag(result.upper)))


def test_affif_constant_triangular_solve_is_exact_affine():
    matrix = jnp.array([[2.0, 0.0], [-1.0, 3.0]])
    rhs = irx.interval(
        jnp.array([[1.0, -2.0], [-3.0, 0.5]]),
        jnp.array([[2.0, 1.0], [4.0, 2.0]]),
    )

    def solve(value):
        return jax.lax.linalg.triangular_solve(
            matrix, value, left_side=True, lower=True
        )

    result = irx.affif(solve)(rhs)
    inverse = jnp.linalg.inv(matrix)
    positive, negative = jnp.maximum(inverse, 0), jnp.minimum(inverse, 0)
    expected_lower = positive @ rhs.lower + negative @ rhs.upper
    expected_upper = positive @ rhs.upper + negative @ rhs.lower
    assert jnp.allclose(result.lower, expected_lower, atol=1e-5)
    assert jnp.allclose(result.upper, expected_upper, atol=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        {
            "left_side": True,
            "lower": True,
            "transpose_a": False,
            "unit_diagonal": False,
        },
        {
            "left_side": True,
            "lower": False,
            "transpose_a": False,
            "unit_diagonal": False,
        },
        {
            "left_side": True,
            "lower": True,
            "transpose_a": True,
            "unit_diagonal": False,
        },
        {
            "left_side": False,
            "lower": True,
            "transpose_a": False,
            "unit_diagonal": False,
        },
        {
            "left_side": True,
            "lower": True,
            "transpose_a": False,
            "unit_diagonal": True,
        },
    ],
)
def test_affif_uncertain_triangular_solve_flags_are_sound(options):
    parameters = irx.interval(
        jnp.array([2.0, 1.5, -0.2, 1.0, -2.0]),
        jnp.array([3.0, 2.5, 0.4, 2.0, 1.0]),
    )

    def solve(p):
        if options["lower"]:
            matrix = jnp.array([[p[0], 0.0], [p[2], p[1]]])
        else:
            matrix = jnp.array([[p[0], p[2]], [0.0, p[1]]])
        rhs = p[3:][None, :] if not options["left_side"] else p[3:, None]
        return jax.lax.linalg.triangular_solve(
            matrix, rhs, conjugate_a=False, **options
        )

    result = irx.affif(solve)(parameters)
    corners = jnp.asarray(
        list(
            itertools.product(
                *zip(parameters.lower.reshape(-1), parameters.upper.reshape(-1))
            )
        )
    )
    values = jax.vmap(solve)(corners)
    assert jnp.all(values >= result.lower - 1e-5)
    assert jnp.all(values <= result.upper + 1e-5)


def test_affif_singular_triangular_solve_returns_top():
    matrix = irx.interval(
        jnp.array([[-1.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, 2.0]]),
    )
    rhs = irx.interval(jnp.ones((2, 1)))
    result = irx.affif(
        lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True, lower=True)
    )(matrix, rhs)
    assert jnp.any(jnp.isneginf(result.lower))
    assert jnp.any(jnp.isposinf(result.upper))


def test_matrix_level_cholesky_is_sound_but_recursive_baseline_is_tighter():
    center = jnp.array([[3.0, 0.15], [0.15, 5.0]])
    matrix = irx.interval(center - 0.1, center + 0.1)
    affine_matrix = irx.interval_to_affine_bound(matrix)

    matrix_result = aif_module._cholesky_matrix(affine_matrix).concretize()
    baseline_result = aif_module._cholesky_recursive(affine_matrix).concretize()
    corners = jnp.asarray(
        list(
            itertools.product(
                *zip(matrix.lower.reshape(-1), matrix.upper.reshape(-1))
            )
        )
    ).reshape((-1, 2, 2))
    values = jax.vmap(jnp.linalg.cholesky)(corners)

    assert jnp.all(values >= matrix_result.lower - 1e-5)
    assert jnp.all(values <= matrix_result.upper + 1e-5)
    assert jnp.sum(baseline_result.width) < jnp.sum(matrix_result.width)
    assert aif_module.affine_inclusion_registry[LA.cholesky_p] is (
        aif_module._cholesky_recursive
    )


def test_matrix_level_triangular_solve_is_sound_but_baseline_is_tighter():
    matrix_center = jnp.array([[2.03, 0.0], [0.09, 3.12]])
    matrix_radius = 0.1 * jnp.tril(jnp.ones((2, 2)))
    matrix = irx.interval(matrix_center - matrix_radius, matrix_center + matrix_radius)
    rhs_center = jnp.array([[-0.3, -0.2], [-0.1, 0.0]])
    rhs = irx.interval(rhs_center - 0.1, rhs_center + 0.1)
    domain_lower = jnp.concatenate((matrix.lower.ravel(), rhs.lower.ravel()))
    domain_upper = jnp.concatenate((matrix.upper.ravel(), rhs.upper.ravel()))
    affine_matrix = irx.interval_to_affine_bound(
        matrix, domain_lower, domain_upper, offset=0
    )
    affine_rhs = irx.interval_to_affine_bound(
        rhs, domain_lower, domain_upper, offset=matrix.size
    )

    options = {"left_side": True, "lower": True}
    matrix_result = aif_module._triangular_solve_matrix(
        affine_matrix, affine_rhs, **options
    ).concretize()
    baseline_result = aif_module._triangular_solve_recursive(
        affine_matrix, affine_rhs, **options
    ).concretize()
    endpoints = list(
        zip(domain_lower.reshape(-1), domain_upper.reshape(-1), strict=True)
    )
    corners = jnp.asarray(list(itertools.product(*endpoints)))
    matrices = corners[:, : matrix.size].reshape((-1, 2, 2))
    right_sides = corners[:, matrix.size :].reshape((-1, 2, 2))
    values = jax.vmap(
        lambda a, b: jax.lax.linalg.triangular_solve(a, b, **options)
    )(matrices, right_sides)

    assert jnp.all(values >= matrix_result.lower - 1e-5)
    assert jnp.all(values <= matrix_result.upper + 1e-5)
    assert jnp.sum(baseline_result.width) < jnp.sum(matrix_result.width)
    assert aif_module.affine_inclusion_registry[LA.triangular_solve_p] is (
        aif_module._triangular_solve_recursive
    )
