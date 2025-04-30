import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.integrators.latent_linear_integrator import LatentLinearIntegrator


@pytest.fixture
def n_latent_dims():
    return 5


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def progressive_iterations():
    return 100


@pytest.fixture
def z(batch_size, n_latent_dims):
    return tf.random.normal((batch_size, n_latent_dims))


@pytest.fixture
def z_conditional(batch_size, n_latent_dims):
    return tf.random.normal((batch_size, n_latent_dims * 2))


@pytest.fixture
def z_hat_conditional(batch_size, n_latent_dims):
    return tf.random.normal((batch_size, n_latent_dims * 2))


@pytest.fixture
def integrator_no_cond(n_latent_dims, progressive_iterations):
    return LatentLinearIntegrator(
        n_latent_dims=n_latent_dims,
        is_conditioned_on_z=False,
        is_conditioned_on_z_hat=False,
        progressive_iterations=progressive_iterations,
    )


@pytest.fixture
def integrator_cond_z(n_latent_dims, progressive_iterations):
    return LatentLinearIntegrator(
        n_latent_dims=n_latent_dims,
        is_conditioned_on_z=True,
        is_conditioned_on_z_hat=False,
        progressive_iterations=progressive_iterations,
    )


@pytest.fixture
def integrator_cond_z_hat(n_latent_dims, progressive_iterations):
    return LatentLinearIntegrator(
        n_latent_dims=n_latent_dims,
        is_conditioned_on_z=False,
        is_conditioned_on_z_hat=True,
        progressive_iterations=progressive_iterations,
    )


@pytest.fixture
def integrator_cond_both(n_latent_dims, progressive_iterations):
    return LatentLinearIntegrator(
        n_latent_dims=n_latent_dims,
        is_conditioned_on_z=True,
        is_conditioned_on_z_hat=True,
        progressive_iterations=progressive_iterations,
    )


def test_init(integrator_no_cond, n_latent_dims, progressive_iterations):
    assert isinstance(integrator_no_cond, LatentLinearIntegrator)
    assert not integrator_no_cond.is_conditioned_on_z
    assert not integrator_no_cond.is_conditioned_on_z_hat
    assert integrator_no_cond.total_iterations == progressive_iterations
    assert integrator_no_cond.r_network.units == n_latent_dims
    assert integrator_no_cond.b_network.units == n_latent_dims


def test_call_no_cond(integrator_no_cond, z, batch_size, n_latent_dims):
    inputs = {Constants.MODEL_OUTPUTS_Z: z}
    # training mode
    z_hat_train = integrator_no_cond(inputs, training=True)
    assert z_hat_train.shape == (batch_size, n_latent_dims)
    # inference mode
    z_hat_infer = integrator_no_cond(inputs, training=False)
    assert z_hat_infer.shape == (batch_size, n_latent_dims)
    # check progressive scaling effect (training vs inference)
    # note: due to random weights, we can't check exact values, but training output should differ from inference
    # reset iteration counter for consistent testing if needed (though not strictly necessary here)
    integrator_no_cond.current_iteration.assign(1.0)
    z_hat_train_again = integrator_no_cond(inputs, training=True)
    # ensure the training output is different from the inference output due to scaling
    # use tf.reduce_any to check if at least one element is different, allowing for float precision issues
    assert tf.reduce_any(tf.abs(z_hat_train_again - z_hat_infer) > 1e-6)


def test_call_cond_z(integrator_cond_z, z, z_conditional, batch_size, n_latent_dims):
    inputs = {
        Constants.MODEL_OUTPUTS_Z: z,
        Constants.MODULE_INPUTS_CONDITIONED_Z: z_conditional,
    }
    z_hat = integrator_cond_z(inputs, training=False)
    assert z_hat.shape == (batch_size, n_latent_dims)


def test_call_cond_z_hat(
    integrator_cond_z_hat, z, z_hat_conditional, batch_size, n_latent_dims
):
    inputs = {
        Constants.MODEL_OUTPUTS_Z: z,
        Constants.MODULE_INPUTS_CONDITIONED_Z_HAT: z_hat_conditional,
    }
    z_hat = integrator_cond_z_hat(inputs, training=False)
    assert z_hat.shape == (batch_size, n_latent_dims)


def test_call_cond_both(
    integrator_cond_both, z, z_conditional, z_hat_conditional, batch_size, n_latent_dims
):
    inputs = {
        Constants.MODEL_OUTPUTS_Z: z,
        Constants.MODULE_INPUTS_CONDITIONED_Z: z_conditional,
        Constants.MODULE_INPUTS_CONDITIONED_Z_HAT: z_hat_conditional,
    }
    z_hat = integrator_cond_both(inputs, training=False)
    assert z_hat.shape == (batch_size, n_latent_dims)


def test_progressive_scaling(integrator_no_cond, z, progressive_iterations):
    inputs = {Constants.MODEL_OUTPUTS_Z: z}
    integrator_no_cond.current_iteration.assign(1.0)

    # call multiple times in training mode
    outputs = []
    for i in range(progressive_iterations + 5):  # Go slightly beyond total iterations
        z_hat = integrator_no_cond(inputs, training=True)
        outputs.append(z_hat)
        # check iteration counter increments correctly until total_iterations
        expected_iteration = min(
            float(i + 1 + 1), float(progressive_iterations)
        )  # +1 for assign_add, +1 because current_iteration starts at 1
        assert integrator_no_cond.current_iteration == expected_iteration

    # Check iteration counter stops at total_iterations
    assert integrator_no_cond.current_iteration == progressive_iterations

    # check that output stabilizes after total_iterations
    # allow for minor floating point differences
    tf.debugging.assert_near(outputs[-1], outputs[-2], rtol=1e-5, atol=1e-5)
    tf.debugging.assert_near(outputs[-2], outputs[-3], rtol=1e-5, atol=1e-5)


def test_missing_cond_z_warning(integrator_cond_z, z):
    inputs = {Constants.MODEL_OUTPUTS_Z: z}
    with pytest.warns(UserWarning):
        integrator_cond_z(inputs, training=False)


def test_missing_cond_z_hat_warning(integrator_cond_z_hat, z):
    inputs = {Constants.MODEL_OUTPUTS_Z: z}
    with pytest.warns(UserWarning):
        integrator_cond_z_hat(inputs, training=False)


def test_missing_cond_both_warning(integrator_cond_both, z):
    inputs = {Constants.MODEL_OUTPUTS_Z: z}
    with pytest.warns(UserWarning):
        integrator_cond_both(inputs, training=False)
