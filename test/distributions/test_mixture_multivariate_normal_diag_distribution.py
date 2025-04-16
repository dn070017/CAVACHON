import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)

BATCH_SIZE = 3
N_COMPONENTS = 5
EVENT_DIMS = 4


@pytest.fixture
def params_tensor():
    return tf.random.normal([BATCH_SIZE, N_COMPONENTS, 1 + EVENT_DIMS * 2])


@pytest.fixture
def params_mapping():
    return {
        "logits": tf.random.normal([BATCH_SIZE, N_COMPONENTS]),
        "loc": tf.random.normal([BATCH_SIZE, N_COMPONENTS, EVENT_DIMS]),
        "scale_diag": tf.exp(tf.random.normal([BATCH_SIZE, N_COMPONENTS, EVENT_DIMS])),
    }


def test_from_parameterizer_output_tensor(params_tensor):
    dist = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
        params_tensor
    )
    assert isinstance(dist, MixtureMultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant([BATCH_SIZE]))
    tf.debugging.assert_equal(dist.event_shape_tensor(), tf.constant([EVENT_DIMS]))

    assert isinstance(dist.mixture_distribution, tfp.distributions.Categorical)
    tf.debugging.assert_equal(
        dist.mixture_distribution.batch_shape_tensor(), tf.constant([BATCH_SIZE])
    )
    tf.debugging.assert_equal(
        dist.mixture_distribution.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    tf.debugging.assert_equal(
        tf.shape(dist.mixture_distribution.logits),
        tf.constant([BATCH_SIZE, N_COMPONENTS]),
    )

    assert isinstance(dist.components_distribution, MultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(
        dist.components_distribution.batch_shape_tensor(),
        tf.constant([BATCH_SIZE, N_COMPONENTS]),
    )
    tf.debugging.assert_equal(
        dist.components_distribution.event_shape_tensor(), tf.constant([EVENT_DIMS])
    )
    tf.debugging.assert_equal(
        tf.shape(dist.components_distribution.loc),
        tf.constant([BATCH_SIZE, N_COMPONENTS, EVENT_DIMS]),
    )


def test_from_parameterizer_output_mapping(params_mapping):
    dist = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
        params_mapping
    )
    assert isinstance(dist, MixtureMultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(dist.batch_shape_tensor(), tf.constant([BATCH_SIZE]))
    tf.debugging.assert_equal(dist.event_shape_tensor(), tf.constant([EVENT_DIMS]))

    assert isinstance(dist.mixture_distribution, tfp.distributions.Categorical)
    tf.debugging.assert_equal(
        dist.mixture_distribution.batch_shape_tensor(), tf.constant([BATCH_SIZE])
    )
    tf.debugging.assert_equal(
        dist.mixture_distribution.event_shape_tensor(), tf.constant([], dtype=tf.int32)
    )
    tf.debugging.assert_equal(
        tf.shape(dist.mixture_distribution.logits),
        tf.constant([BATCH_SIZE, N_COMPONENTS]),
    )

    assert isinstance(dist.components_distribution, MultivariateNormalDiagDistribution)
    tf.debugging.assert_equal(
        dist.components_distribution.batch_shape_tensor(),
        tf.constant([BATCH_SIZE, N_COMPONENTS]),
    )
    tf.debugging.assert_equal(
        dist.components_distribution.event_shape_tensor(), tf.constant([EVENT_DIMS])
    )
    tf.debugging.assert_equal(
        tf.shape(dist.components_distribution.loc),
        tf.constant([BATCH_SIZE, N_COMPONENTS, EVENT_DIMS]),
    )
