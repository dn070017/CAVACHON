import pytest
import tensorflow as tf

from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)

BATCH_SHAPE = [3, 5]
EVENT_DIMS = 4


@pytest.fixture
def params_tensor():
    return tf.random.normal(BATCH_SHAPE + [EVENT_DIMS * 2])


@pytest.fixture
def params_mapping():
    return {
        "loc": tf.random.normal(BATCH_SHAPE + [EVENT_DIMS]),
        "scale_diag": tf.exp(tf.random.normal(BATCH_SHAPE + [EVENT_DIMS])),
    }


def test_from_parameterizer_output_tensor(params_tensor):
    dist = MultivariateNormalDiagDistribution.from_parameterizer_output(params_tensor)
    assert isinstance(dist, MultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant(BATCH_SHAPE))
    tf.debugging.assert_equal(dist.event_shape_tensor(), tf.constant([EVENT_DIMS]))
    tf.debugging.assert_equal(
        tf.shape(dist.loc), tf.constant(BATCH_SHAPE + [EVENT_DIMS])
    )


def test_from_parameterizer_output_mapping(params_mapping):
    dist = MultivariateNormalDiagDistribution.from_parameterizer_output(params_mapping)
    assert isinstance(dist, MultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant(BATCH_SHAPE))
    tf.debugging.assert_equal(dist.event_shape_tensor(), tf.constant([EVENT_DIMS]))
    tf.debugging.assert_equal(
        tf.shape(dist.loc), tf.constant(BATCH_SHAPE + [EVENT_DIMS])
    )
