import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.log_transform import LogTransform


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def pseudocount():
    return 1.0


@pytest.fixture
def log_transform(key, pseudocount):
    return LogTransform(key=key, pseudocount=pseudocount)


def test_init(log_transform, key, pseudocount):
    assert isinstance(log_transform, LogTransform)
    assert log_transform.key == key
    assert log_transform.pseudocount == pseudocount


def test_call_dense(log_transform, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])}
    outputs = log_transform(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([0.0, np.log(2.0), np.log(3.0), np.log(4.0)])
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_near(outputs[key], expected_outputs[key], rtol=1e-6)


def test_call_sparse(log_transform, key, pseudocount):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = log_transform(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(
            tf.convert_to_tensor([0.0, np.log(2.0), np.log(3.0), np.log(4.0)])
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
