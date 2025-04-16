import pytest
import tensorflow as tf

from cavachon.distributions.independent_bernoulli_distribution import (
    IndependentBernoulliDistribution,
)


@pytest.fixture
def params_tensor():
    return tf.random.normal([5, 10])


@pytest.fixture
def params_mapping():
    return {"logits": tf.random.normal([5, 10])}


def test_from_parameterizer_output_tensor(params_tensor):
    dist = IndependentBernoulliDistribution.from_parameterizer_output(params_tensor)
    assert isinstance(dist, IndependentBernoulliDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant([5, 10]))
    tf.debugging.assert_equal(
        dist.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    tf.debugging.assert_equal(tf.shape(dist.logits), tf.constant([5, 10]))


def test_from_parameterizer_output_mapping(params_mapping):
    dist = IndependentBernoulliDistribution.from_parameterizer_output(params_mapping)
    assert isinstance(dist, IndependentBernoulliDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant([5, 10]))
    tf.debugging.assert_equal(
        dist.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    tf.debugging.assert_equal(tf.shape(dist.logits), tf.constant([5, 10]))
