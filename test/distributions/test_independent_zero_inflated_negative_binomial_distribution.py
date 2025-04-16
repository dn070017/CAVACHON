import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.independent_zero_inflated_negative_binomial_distribution import (
    IndependentZeroInflatedNegativeBinomialDistribution,
)

BATCH_SIZE = 5
FEATURE_DIMS = 10


@pytest.fixture
def params_tensor():
    return tf.random.normal([BATCH_SIZE, FEATURE_DIMS * 3])


@pytest.fixture
def params_mapping():
    return {
        "logits": tf.random.normal([BATCH_SIZE, FEATURE_DIMS]),
        "mean": tf.random.uniform([BATCH_SIZE, FEATURE_DIMS], minval=0.1, maxval=10.0),
        "dispersion": tf.random.uniform(
            [BATCH_SIZE, FEATURE_DIMS], minval=0.1, maxval=5.0
        ),
    }


def test_from_parameterizer_output_tensor(params_tensor):
    dist = (
        IndependentZeroInflatedNegativeBinomialDistribution.from_parameterizer_output(
            params_tensor
        )
    )
    assert isinstance(dist, IndependentZeroInflatedNegativeBinomialDistribution)
    tf.debugging.assert_equal(
        dist.batch_shape_tensor(), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )
    tf.debugging.assert_equal(
        dist.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    assert isinstance(dist.cat, tfp.distributions.Categorical)
    tf.debugging.assert_equal(
        tf.shape(dist.cat.probs), tf.constant([BATCH_SIZE, FEATURE_DIMS, 2])
    )
    assert isinstance(dist.components[0], tfp.distributions.Deterministic)
    tf.debugging.assert_equal(
        tf.shape(dist.components[0].loc), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )
    assert isinstance(dist.components[1], tfp.distributions.NegativeBinomial)
    tf.debugging.assert_equal(
        tf.shape(dist.components[1].mean()), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )


def test_from_parameterizer_output_mapping(params_mapping):
    dist = (
        IndependentZeroInflatedNegativeBinomialDistribution.from_parameterizer_output(
            params_mapping
        )
    )
    assert isinstance(dist, IndependentZeroInflatedNegativeBinomialDistribution)
    tf.debugging.assert_equal(
        dist.batch_shape_tensor(), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )
    tf.debugging.assert_equal(
        dist.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    assert isinstance(dist.cat, tfp.distributions.Categorical)
    tf.debugging.assert_equal(
        tf.shape(dist.cat.probs), tf.constant([BATCH_SIZE, FEATURE_DIMS, 2])
    )
    assert isinstance(dist.components[0], tfp.distributions.Deterministic)
    tf.debugging.assert_equal(
        tf.shape(dist.components[0].loc), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )
    assert isinstance(dist.components[1], tfp.distributions.NegativeBinomial)
    tf.debugging.assert_equal(
        tf.shape(dist.components[1].mean()), tf.constant([BATCH_SIZE, FEATURE_DIMS])
    )
