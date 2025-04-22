import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.to_sparse import ToSparse


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def to_sparse(key):
    return ToSparse(key=key)


def test_init(to_sparse, key):
    assert isinstance(to_sparse, ToSparse)
    assert to_sparse.key == key


def test_call_dense(to_sparse, key):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])
    inputs = {key: dense_tensor}
    outputs = to_sparse(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0]))
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


def test_call_sparse(to_sparse, key):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    outputs = to_sparse({key: sparse_tensor})
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0]))
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
