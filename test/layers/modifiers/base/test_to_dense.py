import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.to_dense import ToDense


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def to_dense(key):
    return ToDense(key=key)


def test_init(to_dense, key):
    assert isinstance(to_dense, ToDense)
    assert to_dense.key == key


def test_call_dense(to_dense, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])}
    outputs = to_dense(inputs)
    expected_outputs = {key: tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])}
    assert isinstance(outputs, dict)
    assert key in outputs
    assert isinstance(outputs[key], tf.Tensor)
    print(outputs[key], expected_outputs[key])
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])


def test_call_sparse(to_dense, key):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = to_dense(inputs)
    expected_outputs = {key: tf.convert_to_tensor([0.0, 1.0, 0.0, 3.0])}
    assert isinstance(outputs, dict)
    assert key in outputs
    assert isinstance(outputs[key], tf.Tensor)
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
