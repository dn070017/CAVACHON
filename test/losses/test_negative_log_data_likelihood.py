"""NegativeLogDataLikelihood: ReflectionHandler.get_class_by_name
(cavachon.utils) returns a fake distribution class, and
cavachon.layers.ProgressiveScaler is FakeProgressiveScaler (conftest)."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from unittest.mock import MagicMock

from cavachon.losses.negative_log_data_likelihood import NegativeLogDataLikelihood

from .conftest import make_fake_distribution_class


def test_resolves_distribution_by_name(fakes):
    loss = NegativeLogDataLikelihood("IndependentBernoulli")
    fakes.get_class_by_name.assert_called_once_with(
        "IndependentBernoulli", "distributions", "Distribution"
    )
    assert loss.dist_x_z_class is fakes.bernoulli


def test_accepts_class_directly(fakes):
    cls = MagicMock()
    loss = NegativeLogDataLikelihood(cls)
    fakes.get_class_by_name.assert_not_called()
    assert loss.dist_x_z_class is cls


def test_lookup_error_propagates(fakes):
    fakes.get_class_by_name.side_effect = AssertionError("not found")
    with pytest.raises(AssertionError):
        NegativeLogDataLikelihood("DoesNotExist")


def test_defaults_construct_scaler(fakes):
    loss = NegativeLogDataLikelihood("IndependentBernoulli")
    assert loss.name == "negative_log_data_likelihood"
    assert isinstance(loss.weight, fakes.scaler_cls)
    assert loss.weight.total_iterations == 1
    assert loss.weight.scale == 1.0


def _bernoulli_case():
    logits = tf.constant([[-1.0, 0.0, 2.0], [0.5, -2.0, 1.0]])
    x = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    per_sample = -tf.reduce_sum(
        tfp.distributions.Bernoulli(logits=logits).log_prob(x), -1
    )
    return logits, x, per_sample.numpy()


def test_distribution_built_from_y_pred_and_evaluated_on_y_true(fakes):
    logits, x, _ = _bernoulli_case()
    dist = MagicMock()
    dist.log_prob.return_value = tf.zeros((2, 3))
    fakes.bernoulli.from_parameterizer_output.side_effect = None
    fakes.bernoulli.from_parameterizer_output.return_value = dist
    NegativeLogDataLikelihood("IndependentBernoulli").call(x, logits)
    np.testing.assert_array_equal(
        fakes.bernoulli.from_parameterizer_output.call_args.args[0], logits
    )
    np.testing.assert_array_equal(dist.log_prob.call_args.args[0], x)


def test_sums_over_event_axis_and_negates(fakes):
    dist = MagicMock()
    dist.log_prob.return_value = tf.constant([[-1.0, -2.0], [-0.5, -0.5]])
    fakes.bernoulli.from_parameterizer_output.side_effect = None
    fakes.bernoulli.from_parameterizer_output.return_value = dist
    loss = NegativeLogDataLikelihood("IndependentBernoulli")
    np.testing.assert_allclose(
        loss.call(tf.zeros((2, 2)), tf.zeros((2, 2))), [3.0, 1.0]
    )
    np.testing.assert_allclose(loss(tf.zeros((2, 2)), tf.zeros((2, 2))), 2.0)


def test_bernoulli_per_sample_and_reduced():
    logits, x, ref = _bernoulli_case()
    loss = NegativeLogDataLikelihood("IndependentBernoulli")
    np.testing.assert_allclose(loss.call(x, logits), ref, rtol=1e-5)
    np.testing.assert_allclose(loss(x, logits), ref.mean(), rtol=1e-5)


def test_output_scaled_by_scaler_output():
    logits, x, ref = _bernoulli_case()
    loss = NegativeLogDataLikelihood("IndependentBernoulli", weight=0.5)
    assert loss.weight.scale == 0.5
    np.testing.assert_allclose(loss(x, logits), 0.5 * ref.mean(), rtol=1e-5)
    loss.weight.pin_to(0.0)
    assert float(loss(x, logits)) == 0.0


def test_sample_weight():
    logits, x, ref = _bernoulli_case()
    loss = NegativeLogDataLikelihood("IndependentBernoulli")
    sw = tf.constant([1.0, 0.0])
    np.testing.assert_allclose(loss(x, logits, sample_weight=sw), ref[0] / 2, rtol=1e-5)


def test_vector_event_distribution_returns_batch_mean(fakes):
    def from_params(params, **kwargs):
        loc, scale = tf.split(params, 2, axis=-1)
        return tfp.distributions.MultivariateNormalDiag(loc, scale)

    fakes.get_class_by_name.return_value = make_fake_distribution_class(from_params)
    loc, scale = tf.zeros((4, 2)), tf.ones((4, 2))
    y_pred = tf.concat([loc, scale], -1)
    x = tf.zeros((4, 2))
    ref = -tfp.distributions.MultivariateNormalDiag(loc, scale).log_prob(x).numpy()
    loss = NegativeLogDataLikelihood("MultivariateNormalDiag")
    assert loss.call(x, y_pred).shape == (4,)
    np.testing.assert_allclose(loss(x, y_pred), ref.mean(), rtol=1e-5)
