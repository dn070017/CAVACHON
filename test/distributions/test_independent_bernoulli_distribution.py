import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.independent_bernoulli_distribution import (
    IndependentBernoulliDistribution,
)


def _logits():
    return tf.constant([[-2.0, 0.0, 3.0], [1.0, -1.0, 0.5]])


def test_from_tensor_shapes_and_probs():
    logits = _logits()
    d = IndependentBernoulliDistribution.from_parameterizer_output(logits)
    assert d.batch_shape == (2, 3)
    assert d.event_shape == ()
    np.testing.assert_allclose(d.probs_parameter(), tf.sigmoid(logits), rtol=1e-6)


def test_from_mapping_matches_tensor():
    logits = _logits()
    a = IndependentBernoulliDistribution.from_parameterizer_output(logits)
    b = IndependentBernoulliDistribution.from_parameterizer_output({"logits": logits})
    x = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(a.log_prob(x), b.log_prob(x))


def test_log_prob_matches_reference():
    logits = _logits()
    x = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    d = IndependentBernoulliDistribution.from_parameterizer_output(logits)
    p = 1 / (1 + np.exp(-logits.numpy()))
    ref = x.numpy() * np.log(p) + (1 - x.numpy()) * np.log(1 - p)
    np.testing.assert_allclose(d.log_prob(x), ref, rtol=1e-5)


def test_large_logits_are_stable():
    logits = tf.constant([[-200.0, 200.0]])
    d = IndependentBernoulliDistribution.from_parameterizer_output(logits)
    lp = d.log_prob(tf.constant([[1.0, 0.0]])).numpy()
    assert np.all(np.isfinite(lp))
    np.testing.assert_allclose(lp, [[-200.0, -200.0]], rtol=1e-5)


def test_direct_constructor():
    d = IndependentBernoulliDistribution(logits=tf.zeros((2,)))
    assert isinstance(d, tfp.distributions.Bernoulli)
    np.testing.assert_allclose(d.mean(), [0.5, 0.5])


@pytest.mark.parametrize(
    "make_params",
    [lambda: np.zeros((2, 3), np.float32), lambda: tf.Variable(tf.zeros((2, 3)))],
    ids=["numpy", "tf.Variable"],
)
def test_non_tensor_input_accepted(make_params):
    # regression: numpy / tf.Variable params used to raise UnboundLocalError.
    d = IndependentBernoulliDistribution.from_parameterizer_output(make_params())
    assert d.batch_shape == (2, 3)
    np.testing.assert_allclose(d.probs_parameter(), 0.5)


def test_returns_instance_of_cls():
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    d = IndependentBernoulliDistribution.from_parameterizer_output(tf.zeros((2, 3)))
    assert isinstance(d, IndependentBernoulliDistribution)


def test_kwargs_forwarded():
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    d = IndependentBernoulliDistribution.from_parameterizer_output(
        tf.zeros((2, 3)), validate_args=True, name="my_bern"
    )
    assert d.validate_args
    assert d.name == "my_bern"
