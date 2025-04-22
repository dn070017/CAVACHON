import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.exp_transform import ExpTransform


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def exp_transform(key):
    return ExpTransform(key=key)


def test_init(exp_transform, key):
    assert isinstance(exp_transform, ExpTransform)
    assert exp_transform.key == key


def test_call_dense(exp_transform, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])}
    outputs = exp_transform(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([1.0, np.exp(1.0), np.exp(2.0), np.exp(3.0)])
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_near(outputs[key], expected_outputs[key], rtol=1e-6)


def test_call_sparse(exp_transform, key):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = exp_transform(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(
            tf.convert_to_tensor([1.0, np.exp(1.0), np.exp(2.0), np.exp(3.0)])
        )
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert isinstance(outputs[key], tf.SparseTensor)
    tf.debugging.assert_equal(outputs[key].indices, expected_outputs[key].indices)
    tf.debugging.assert_near(
        outputs[key].values, expected_outputs[key].values, rtol=1e-6
    )
    tf.debugging.assert_equal(
        outputs[key].dense_shape, expected_outputs[key].dense_shape
    )
