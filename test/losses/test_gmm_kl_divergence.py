"""GMMKLDivergence: cavachon.layers.ProgressiveScaler and the
cavachon.distributions classes (MultivariateNormalDiag /
MixtureMultivariateNormalDiag) are replaced by fakes from conftest that
build plain tfp distributions."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import logsumexp, softmax

from cavachon.losses.gmm_kl_divergence import GMMKLDivergence


def _inputs(batch=5, k=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(1, k)).astype(np.float32)
    p_loc = rng.normal(size=(1, k, d)).astype(np.float32) * 2
    p_scale = rng.uniform(0.5, 1.5, size=(1, k, d)).astype(np.float32)
    y_true = np.concatenate([logits[..., None], p_loc, p_scale], -1)

    q_loc = rng.normal(size=(batch, d)).astype(np.float32)
    q_scale = rng.uniform(0.3, 1.0, size=(batch, d)).astype(np.float32)
    z = (q_loc + q_scale * rng.normal(size=(batch, d))).astype(np.float32)
    y_pred = np.concatenate([z, q_loc, q_scale], -1)
    return dict(
        logits=logits, p_loc=p_loc, p_scale=p_scale,
        y_true=tf.constant(y_true),
        q_loc=q_loc, q_scale=q_scale, z=z, y_pred=tf.constant(y_pred),
    )


def _np_gauss_logpdf(x, loc, scale):
    return np.sum(
        -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * ((x - loc) / scale) ** 2,
        -1,
    )


def _np_reference(inp):
    """Single-sample estimate of KL(q(z, y | x) || p(z, y)) with
    q(y | z) = p(y | z) (VaDE / MFCVAE)."""
    z = inp["z"].astype(np.float64)
    logqz_x = _np_gauss_logpdf(z, inp["q_loc"], inp["q_scale"])
    logpz_y = _np_gauss_logpdf(z[:, None], inp["p_loc"], inp["p_scale"])  # (B,K)
    logpy = inp["logits"] - logsumexp(inp["logits"], -1, keepdims=True)
    joint = logpz_y + logpy
    r = softmax(joint, -1)
    return logqz_x + np.sum(r * (np.log(r) - joint), -1)


def test_defaults_construct_scaler(fakes):
    loss = GMMKLDivergence()
    assert loss.name == "gmm_kl_divergence"
    assert isinstance(loss.weight, fakes.scaler_cls)
    assert loss.weight.total_iterations == 1
    assert loss.weight.scale == 1.0


def test_distributions_built_from_expected_slices(fakes):
    inp = _inputs()
    GMMKLDivergence().call(inp["y_true"], inp["y_pred"])
    # q(z|x) from y_pred[..., d:], p(z|y) from y_true[..., 1:], p(z) from y_true
    mvn_args = [c.args[0] for c in fakes.mvn.from_parameterizer_output.call_args_list]
    assert len(mvn_args) == 2
    np.testing.assert_array_equal(mvn_args[0], inp["y_pred"][..., 2:])
    np.testing.assert_array_equal(mvn_args[1], inp["y_true"][..., 1:])
    fakes.mixture.from_parameterizer_output.assert_called_once()
    np.testing.assert_array_equal(
        fakes.mixture.from_parameterizer_output.call_args.args[0], inp["y_true"]
    )


def test_per_sample_matches_numpy_reference():
    inp = _inputs()
    out = GMMKLDivergence().call(inp["y_true"], inp["y_pred"])
    assert out.shape == (5,)
    np.testing.assert_allclose(out, _np_reference(inp), rtol=1e-4, atol=1e-4)


def test_equals_logq_minus_log_marginal_prior():
    # Analytical identity: the estimate collapses to log q(z|x) - log p(z).
    inp = _inputs(seed=3)
    prior = tfp.distributions.MixtureSameFamily(
        tfp.distributions.Categorical(logits=inp["logits"][0]),
        tfp.distributions.MultivariateNormalDiag(inp["p_loc"][0], inp["p_scale"][0]),
    )
    q = tfp.distributions.MultivariateNormalDiag(inp["q_loc"], inp["q_scale"])
    ref = q.log_prob(inp["z"]) - prior.log_prob(inp["z"])
    out = GMMKLDivergence().call(inp["y_true"], inp["y_pred"])
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_call_reduces_to_batch_mean_and_is_scaled_by_scaler():
    inp = _inputs()
    ref = _np_reference(inp).mean()
    loss = GMMKLDivergence(weight=3.0)
    assert loss.weight.scale == 3.0
    np.testing.assert_allclose(loss(inp["y_true"], inp["y_pred"]), 3.0 * ref, rtol=1e-4)
    loss.weight.value = 0.5
    np.testing.assert_allclose(loss(inp["y_true"], inp["y_pred"]), 0.5 * ref, rtol=1e-4)


def test_zero_scaler_output_disables_loss():
    inp = _inputs()
    loss = GMMKLDivergence(weight=2.0)
    loss.weight.pin_to(0.0)
    assert float(loss(inp["y_true"], inp["y_pred"])) == 0.0


def test_single_component_equals_standard_mc_kl():
    inp = _inputs(k=1)
    q = tfp.distributions.MultivariateNormalDiag(inp["q_loc"], inp["q_scale"])
    p = tfp.distributions.MultivariateNormalDiag(inp["p_loc"][0, 0], inp["p_scale"][0, 0])
    ref = q.log_prob(inp["z"]) - p.log_prob(inp["z"])
    out = GMMKLDivergence().call(inp["y_true"], inp["y_pred"])
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_far_away_z_is_finite():
    inp = _inputs()
    yp = inp["y_pred"].numpy().copy()
    yp[:, :2] = 50.0
    out = GMMKLDivergence().call(inp["y_true"], tf.constant(yp)).numpy()
    assert np.all(np.isfinite(out))


def test_gradients_finite():
    inp = _inputs()
    v, w = inp["y_pred"], inp["y_true"]
    with tf.GradientTape() as tape:
        tape.watch([v, w])
        out = GMMKLDivergence()(w, v)
    gv, gw = tape.gradient(out, [v, w])
    assert np.all(np.isfinite(gv.numpy())) and np.all(np.isfinite(gw.numpy()))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: event_dims computed from y_pred.shape[1] (axis 1) while "
    "slicing on the last axis; rank-3 y_pred is split incorrectly",
)
def test_rank3_y_pred_uses_last_axis():
    inp = _inputs(batch=4, d=2)
    ref = _np_reference(inp)
    yp3 = tf.stack([inp["y_pred"]] * 5, axis=1)  # (4, 5, 6)
    out = GMMKLDivergence().call(inp["y_true"][:, None], yp3).numpy()
    np.testing.assert_allclose(out, np.stack([ref] * 5, 1), rtol=1e-4, atol=1e-4)
