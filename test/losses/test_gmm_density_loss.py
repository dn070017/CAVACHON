"""GMMDensityLoss: cavachon.distributions.MixtureMultivariateNormalDiag
is replaced by a fake (conftest) that builds a tfp MixtureSameFamily;
the ProgressiveScaler weight is a FakeProgressiveScaler instance."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from unittest.mock import MagicMock

from cavachon.losses.gmm_density_loss import GMMDensityLoss


def _inputs(batch=4, k=3, d=2, prior_batch=None, seed=0):
    rng = np.random.default_rng(seed)
    pb = batch if prior_batch is None else prior_batch
    logits = rng.normal(size=(pb, k)).astype(np.float32)
    loc = rng.normal(size=(pb, k, d)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, size=(pb, k, d)).astype(np.float32)
    y_true = tf.constant(np.concatenate([logits[..., None], loc, scale], -1))
    z = tf.constant(rng.normal(size=(batch, d)).astype(np.float32))
    ref = tfp.distributions.MixtureSameFamily(
        tfp.distributions.Categorical(logits=logits),
        tfp.distributions.MultivariateNormalDiag(loc, scale),
    )
    return y_true, z, ref


def test_defaults():
    loss = GMMDensityLoss()
    assert loss.name == "gmm_density_loss"
    assert loss.weight is None


def test_prior_built_from_y_true(fakes):
    y_true, z, _ = _inputs()
    GMMDensityLoss()(y_true, z)
    fakes.mixture.from_parameterizer_output.assert_called_once()
    np.testing.assert_array_equal(
        fakes.mixture.from_parameterizer_output.call_args.args[0], y_true
    )


def test_is_negative_mean_of_controlled_log_prob(fakes):
    dist = MagicMock()
    dist.log_prob.return_value = tf.constant([-1.0, -2.0, -3.0, -6.0])
    fakes.mixture.from_parameterizer_output.side_effect = None
    fakes.mixture.from_parameterizer_output.return_value = dist
    z = tf.zeros((4, 2))
    out = GMMDensityLoss()(tf.zeros((4, 3, 5)), z)
    assert out.shape == ()
    np.testing.assert_allclose(out, 3.0)
    np.testing.assert_array_equal(dist.log_prob.call_args.args[0], z)


def test_unweighted_is_negative_mean_log_prob():
    y_true, z, ref = _inputs()
    out = GMMDensityLoss()(y_true, z)
    np.testing.assert_allclose(out, -np.mean(ref.log_prob(z)), rtol=1e-5)


def test_shared_prior_broadcasts():
    y_true, z, ref = _inputs(batch=6, prior_batch=1)
    out = GMMDensityLoss()(y_true, z)
    np.testing.assert_allclose(out, -np.mean(ref.log_prob(z)), rtol=1e-5)


def test_tf_variable_weight():
    y_true, z, ref = _inputs()
    w = tf.Variable(2.0, trainable=False)
    loss = GMMDensityLoss(weight=w)
    np.testing.assert_allclose(loss(y_true, z), -2.0 * np.mean(ref.log_prob(z)), rtol=1e-5)
    w.assign(0.0)
    np.testing.assert_allclose(loss(y_true, z), 0.0, atol=1e-7)


def test_gradient_wrt_prior_params_and_z():
    y_true, z, _ = _inputs()
    with tf.GradientTape() as tape:
        tape.watch([y_true, z])
        out = GMMDensityLoss()(y_true, z)
    gyt, gz = tape.gradient(out, [y_true, z])
    assert np.all(np.isfinite(gyt.numpy())) and np.all(np.isfinite(gz.numpy()))
    assert np.any(gz.numpy() != 0)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: docstring allows a ProgressiveScaler weight, but call() "
    "does `loss * self.weight` (not self.weight(...)); a callable, "
    "non-tensor weight such as a Keras Layer cannot be multiplied",
)
def test_progressive_scaler_weight_supported(fakes):
    y_true, z, ref = _inputs()
    scaler = fakes.scaler_cls(total_iterations=1, scale=2.0)
    loss = GMMDensityLoss(weight=scaler)
    np.testing.assert_allclose(loss(y_true, z), -2.0 * np.mean(ref.log_prob(z)), rtol=1e-5)
