import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

from cavachon.distributions.independent_zero_inflated_negative_binomial_distribution import (
    IndependentZeroInflatedNegativeBinomialDistribution as ZINB,
)


def _params():
    logits = np.array([[-1.0, 0.0, 2.0], [0.5, -3.0, 1.0]], np.float32)
    mean = np.array([[2.0, 5.0, 0.5], [10.0, 1.0, 3.0]], np.float32)
    disp = np.array([[1.0, 0.5, 2.0], [0.2, 1.0, 4.0]], np.float32)
    return logits, mean, disp


def _reference_log_prob(x, logits, mean, disp):
    """Zero-inflated NB with dropout prob pi = sigmoid(logits).

    tfp's experimental_from_mean_dispersion uses total_count = 1/disp,
    i.e. var = mean + disp * mean**2.
    """
    pi = 1 / (1 + np.exp(-logits.astype(np.float64)))
    r = 1.0 / disp.astype(np.float64)
    p = r / (r + mean.astype(np.float64))  # scipy success prob
    nb = stats.nbinom.pmf(x, r, p)
    zero = (x == 0).astype(np.float64)
    return np.log(pi * zero + (1 - pi) * nb)


def test_from_tensor_shapes():
    logits, mean, disp = _params()
    d = ZINB.from_parameterizer_output(tf.concat([logits, mean, disp], -1))
    assert isinstance(d, ZINB)
    assert isinstance(d, tfp.distributions.Mixture)
    assert d.batch_shape == (2, 3)
    assert d.event_shape == ()


def test_mapping_matches_tensor():
    logits, mean, disp = _params()
    a = ZINB.from_parameterizer_output(tf.concat([logits, mean, disp], -1))
    b = ZINB.from_parameterizer_output(
        {"logits": logits, "mean": mean, "dispersion": disp}
    )
    x = tf.constant([[0.0, 3.0, 1.0], [7.0, 0.0, 2.0]])
    np.testing.assert_allclose(a.log_prob(x), b.log_prob(x), rtol=1e-6)


def test_log_prob_matches_scipy_reference():
    logits, mean, disp = _params()
    d = ZINB.from_parameterizer_output(tf.concat([logits, mean, disp], -1))
    x = np.array([[0.0, 3.0, 1.0], [7.0, 0.0, 2.0]], np.float32)
    ref = _reference_log_prob(x, logits, mean, disp)
    np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-4, atol=1e-5)


def test_mean_is_one_minus_dropout_times_mean():
    logits, mean, disp = _params()
    d = ZINB.from_parameterizer_output(tf.concat([logits, mean, disp], -1))
    pi = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(d.mean(), (1 - pi) * mean + pi * 1e-7, rtol=1e-5)


def test_sample_shape_and_nonnegative_integers():
    logits, mean, disp = _params()
    d = ZINB.from_parameterizer_output(tf.concat([logits, mean, disp], -1))
    s = d.sample(4, seed=1).numpy()
    assert s.shape == (4, 2, 3)
    assert np.all(s >= 0)
    np.testing.assert_array_equal(s, np.round(s))


def test_zero_mean_component_concentrates_at_zero():
    # Very large logits -> almost surely zero.
    d = ZINB.from_parameterizer_output(
        {
            "logits": tf.constant([[30.0]]),
            "mean": tf.constant([[5.0]]),
            "dispersion": tf.constant([[1.0]]),
        }
    )
    np.testing.assert_allclose(d.log_prob(tf.zeros((1, 1))), [[0.0]], atol=1e-5)


def test_kwargs_forwarded():
    logits, mean, disp = _params()
    d = ZINB.from_parameterizer_output(
        tf.concat([logits, mean, disp], -1), name="zinb_test"
    )
    assert d.name == "zinb_test"


def test_last_dim_not_multiple_of_three_raises():
    with pytest.raises(Exception):
        ZINB.from_parameterizer_output(tf.ones((2, 4)))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: dropout probs are built as sigmoid(logits) then 1-probs, "
    "so for moderately large logits 1-probs underflows to 0 in float32 and "
    "the NB component is dropped entirely (should use logits / log_sigmoid)",
)
def test_large_logit_nonzero_count_log_prob_is_accurate():
    logits = np.array([[20.0]], np.float32)
    mean = np.array([[5.0]], np.float32)
    disp = np.array([[1.0]], np.float32)
    d = ZINB.from_parameterizer_output(
        {"logits": logits, "mean": mean, "dispersion": disp}
    )
    x = np.array([[3.0]], np.float32)
    ref = _reference_log_prob(x, logits, mean, disp)  # ~ -20 + log NB(3)
    np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-3)
