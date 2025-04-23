import pytest
import tensorflow as tf

from cavachon.layers.parameterizers.multivariate_normal_diag_sampler import (
    MultivariateNormalDiagSampler,
)


@pytest.fixture
def sampler():
    return MultivariateNormalDiagSampler()


def test_multivariate_normal_diag_sampler_train_mode(sampler):
    batch_size = 5
    input_dim = 10
    inputs = tf.random.normal(shape=(batch_size, 2 * input_dim))  # loc and scale_diag
    output = sampler(inputs, training=True)
    assert output.shape == (batch_size, input_dim)


def test_multivariate_normal_diag_sampler_test_mode(sampler):
    batch_size = 5
    input_dim = 10
    inputs = tf.random.normal(shape=(batch_size, 2 * input_dim))
    output = sampler(inputs, training=False)
    assert output.shape == (batch_size, input_dim)


def test_multivariate_normal_diag_sampler_output_value_test_mode(sampler):
    batch_size = 5
    input_dim = 10
    inputs = tf.random.normal(shape=(batch_size, 2 * input_dim))
    loc, _ = tf.split(inputs, 2, axis=-1)
    output = sampler(inputs, training=False)
    tf.debugging.assert_equal(output, loc)
