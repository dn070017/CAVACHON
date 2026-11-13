"""StandardKLDivergence: only cross-package dependency is
cavachon.layers.ProgressiveScaler, replaced by FakeProgressiveScaler
(see conftest.fakes)."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.losses.standard_kl_divergence import StandardKLDivergence


def _y_pred(batch=4, d=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(batch, d)).astype(np.float32)
    mean = rng.normal(size=(batch, d)).astype(np.float32)
    std = rng.uniform(0.3, 2.0, size=(batch, d)).astype(np.float32)
    return z, mean, std, np.concatenate([z, mean, std], -1)


def _tfp_kl(mean, std):
    q = tfp.distributions.MultivariateNormalDiag(mean, std)
    p = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mean), tf.ones_like(std))
    return tfp.distributions.kl_divergence(q, p).numpy()


def test_defaults_construct_scaler(fakes):
    loss = StandardKLDivergence()
    assert loss.name == "standard_kl_divergence"
    assert isinstance(loss.weight, fakes.scaler_cls)
    assert loss.weight.total_iterations == 1
    assert loss.weight.scale == 1.0


def test_weight_passed_to_scaler_as_float(fakes):
    loss = StandardKLDivergence(weight=3)
    assert loss.weight.scale == 3.0 and isinstance(loss.weight.scale, float)


def test_per_sample_kl_matches_tfp():
    _, mean, std, yp = _y_pred()
    per_sample = StandardKLDivergence().call(None, tf.constant(yp)).numpy()
    assert per_sample.shape == (4,)
    np.testing.assert_allclose(per_sample, _tfp_kl(mean, std), rtol=1e-5)


def test_call_reduces_by_batch_mean():
    _, mean, std, yp = _y_pred()
    out = StandardKLDivergence()(tf.zeros((4, 3)), tf.constant(yp)).numpy()
    np.testing.assert_allclose(out, _tfp_kl(mean, std).mean(), rtol=1e-5)


def test_sampled_z_and_y_true_are_ignored():
    _, _, _, yp = _y_pred()
    yp2 = yp.copy()
    yp2[:, :3] = 1e3
    loss = StandardKLDivergence()
    a = loss(tf.zeros((4, 3)), tf.constant(yp))
    b = loss(tf.ones((4, 3)) * 7, tf.constant(yp2))
    np.testing.assert_allclose(a, b)


def test_zero_when_posterior_equals_prior():
    yp = np.concatenate(
        [np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 3))], -1
    ).astype(np.float32)
    np.testing.assert_allclose(
        StandardKLDivergence()(tf.zeros((2, 3)), yp), 0.0, atol=1e-6
    )


def test_nonnegative_and_finite_at_min_std():
    yp = np.concatenate(
        [np.zeros((2, 2)), np.zeros((2, 2)), np.full((2, 2), 1e-7)], -1
    ).astype(np.float32)
    out = StandardKLDivergence().call(None, tf.constant(yp)).numpy()
    assert np.all(np.isfinite(out)) and np.all(out >= 0)


def test_output_is_scaled_by_scaler_output():
    _, mean, std, yp = _y_pred()
    loss = StandardKLDivergence(weight=1.0)
    loss.weight.value = 2.5
    out = loss(tf.zeros((4, 3)), tf.constant(yp)).numpy()
    np.testing.assert_allclose(out, 2.5 * _tfp_kl(mean, std).mean(), rtol=1e-5)
    # the scaler is invoked on a scalar ones tensor
    assert loss.weight.call_args[-1].shape == ()
    np.testing.assert_allclose(loss.weight.call_args[-1], 1.0)


def test_zero_scaler_output_gives_zero_loss():
    _, _, _, yp = _y_pred()
    loss = StandardKLDivergence()
    loss.weight.pin_to(0.0)
    assert float(loss(tf.zeros((4, 3)), tf.constant(yp))) == 0.0


def test_gradient_flows_to_mean_and_std():
    _, _, _, yp = _y_pred()
    v = tf.Variable(yp)
    with tf.GradientTape() as tape:
        out = StandardKLDivergence()(tf.zeros((4, 3)), v)
    g = tape.gradient(out, v).numpy()
    assert np.all(g[:, :3] == 0)
    assert np.any(g[:, 3:] != 0)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: event_dims is taken from y_pred.shape[1] while slicing is "
    "on the last axis, so rank-3 inputs (batch, T, 3d) are split wrongly",
)
def test_rank3_input_uses_last_axis():
    _, mean, std, yp = _y_pred(batch=4, d=2)
    yp3 = np.stack([yp] * 5, axis=1)  # (4, 5, 6)
    out = StandardKLDivergence().call(None, tf.constant(yp3)).numpy()
    np.testing.assert_allclose(
        out, np.stack([_tfp_kl(mean, std)] * 5, axis=1), rtol=1e-5
    )
