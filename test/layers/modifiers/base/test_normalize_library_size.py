import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.normalize_library_size import NormalizeLibrarySize


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def libsize_key(key):
    return f"{key}_{Constants.TENSOR_NAME_LIBSIZE}"


@pytest.fixture
def normalize_library_size(key):
    return NormalizeLibrarySize(key=key)


def test_init(normalize_library_size, key):
    assert isinstance(normalize_library_size, NormalizeLibrarySize)
    assert normalize_library_size.key == key


def test_call_dense(normalize_library_size, key, libsize_key):
    inputs = {key: tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]])}
    outputs = normalize_library_size(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]]),
        libsize_key: tf.convert_to_tensor([50.0], dtype=tf.float32),
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])


def test_call_sparse(normalize_library_size, key, libsize_key):
    dense_tensor = tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = normalize_library_size(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])),
        libsize_key: tf.convert_to_tensor([50.0], dtype=tf.float32),
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    assert isinstance(outputs[key], tf.SparseTensor)

    tf.debugging.assert_equal(outputs[key].indices, expected_outputs[key].indices)
    tf.debugging.assert_equal(outputs[key].values, expected_outputs[key].values)
    tf.debugging.assert_equal(
        outputs[key].dense_shape, expected_outputs[key].dense_shape
    )
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])


def test_call_dense_existing_libsize(normalize_library_size, key, libsize_key):
    inputs = {
        key: tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]]),
        libsize_key: tf.convert_to_tensor([100.0], dtype=tf.float32),
    }
    outputs = normalize_library_size(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([[0.05, 0.05, 0.1, 0.2, 0.1]]),
        libsize_key: tf.convert_to_tensor([100.0], dtype=tf.float32),
    }
    print(outputs, expected_outputs)
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])


def test_call_sparse_existing_libsize(normalize_library_size, key, libsize_key):
    dense_tensor = tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {
        key: sparse_tensor,
        libsize_key: tf.convert_to_tensor([100.0], dtype=tf.float32),
    }
    outputs = normalize_library_size(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([[0.05, 0.05, 0.1, 0.2, 0.1]])),
        libsize_key: tf.convert_to_tensor([100.0], dtype=tf.float32),
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    assert isinstance(outputs[key], tf.SparseTensor)

    tf.debugging.assert_equal(outputs[key].indices, expected_outputs[key].indices)
    tf.debugging.assert_equal(outputs[key].values, expected_outputs[key].values)
    tf.debugging.assert_equal(
        outputs[key].dense_shape, expected_outputs[key].dense_shape
    )
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])
