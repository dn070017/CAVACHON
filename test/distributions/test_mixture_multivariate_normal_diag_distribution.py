import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import logsumexp

from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution as MixMVN,
)


def _params(batch=3, k=4, d=2, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(batch, k)).astype(np.float32)
    loc = rng.normal(size=(batch, k, d)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, size=(batch, k, d)).astype(np.float32)
    packed = np.concatenate([logits[..., None], loc, scale], -1)
    return logits, loc, scale, packed


def _np_log_prob(x, logits, loc, scale):
    # x: (batch, d)
    x = x[:, None, :]
    comp = np.sum(
        -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * ((x - loc) / scale) ** 2,
        -1,
    )
    logw = logits - logsumexp(logits, -1, keepdims=True)
    return logsumexp(logw + comp, -1)


def test_from_tensor_shapes():
    *_, packed = _params()
    dist = MixMVN.from_parameterizer_output(tf.constant(packed))
    assert isinstance(dist, MixMVN)
    assert dist.batch_shape == (3,)
    assert dist.event_shape == (2,)
    assert dist.mixture_distribution.batch_shape == (3,)
    assert dist.components_distribution.batch_shape == (3, 4)


def test_log_prob_matches_numpy_reference():
    logits, loc, scale, packed = _params()
    dist = MixMVN.from_parameterizer_output(tf.constant(packed))
    x = np.random.default_rng(1).normal(size=(3, 2)).astype(np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x), _np_log_prob(x, logits, loc, scale), rtol=1e-5
    )


def test_mapping_matches_tensor():
    logits, loc, scale, packed = _params()
    a = MixMVN.from_parameterizer_output(tf.constant(packed))
    b = MixMVN.from_parameterizer_output(
        {"logits": logits, "loc": loc, "scale_diag": scale}
    )
    x = tf.zeros((3, 2))
    np.testing.assert_allclose(a.log_prob(x), b.log_prob(x), rtol=1e-6)


def test_single_component_reduces_to_mvn():
    logits, loc, scale, packed = _params(batch=2, k=1, d=3)
    dist = MixMVN.from_parameterizer_output(tf.constant(packed))
    ref = tfp.distributions.MultivariateNormalDiag(loc[:, 0], scale[:, 0])
    x = tf.ones((2, 3))
    np.testing.assert_allclose(dist.log_prob(x), ref.log_prob(x), rtol=1e-6)


def test_prior_broadcasts_against_batch_of_samples():
    # Shape used by losses: prior with batch_shape (1,), samples (batch, d).
    *_, packed = _params(batch=1, k=3, d=2)
    dist = MixMVN.from_parameterizer_output(tf.constant(packed))
    lp = dist.log_prob(tf.zeros((5, 2)))
    assert lp.shape == (5,)


def test_mean_is_weighted_component_means():
    logits, loc, scale, packed = _params()
    dist = MixMVN.from_parameterizer_output(tf.constant(packed))
    w = np.exp(logits - logsumexp(logits, -1, keepdims=True))
    np.testing.assert_allclose(
        dist.mean(), np.sum(w[..., None] * loc, -2), rtol=1e-5, atol=1e-6
    )


def test_kwargs_forwarded():
    *_, packed = _params()
    dist = MixMVN.from_parameterizer_output(tf.constant(packed), name="gmm")
    assert dist.name == "gmm"


def test_even_last_dim_raises():
    # last dim must be 2*p + 1
    with pytest.raises(Exception):
        MixMVN.from_parameterizer_output(tf.ones((1, 3, 4)))


def test_tf_variable_params_accepted():
    # regression: tf.Variable params used to raise UnboundLocalError.
    logits, loc, scale, packed = _params()
    dist = MixMVN.from_parameterizer_output(tf.Variable(packed))
    x = np.zeros((3, 2), np.float32)
    np.testing.assert_allclose(
        dist.log_prob(x), _np_log_prob(x, logits, loc, scale), rtol=1e-5
    )
