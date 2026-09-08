import itertools

import jax
import jax.numpy as jnp
import pytest

import immrax as irx


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


def test_affif_tensor_operations_and_uncertain_dot_product():
    box = irx.interval(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))

    def function(x):
        y = jnp.concatenate((x, x[::-1]))
        return jnp.array([1.0, -2.0, 0.5, 3.0]) @ y + x @ x

    result = irx.affif(function)(box)
    assert result.shape == ()
    _check_samples(function, box, result)


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


def test_affif_intentionally_excludes_cholesky():
    matrix = irx.interval(jnp.eye(2), 2 * jnp.eye(2))
    with pytest.raises(NotImplementedError, match="cholesky"):
        irx.affif(jnp.linalg.cholesky)(matrix)
