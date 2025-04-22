import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.scale_library_size import ScaleLibrarySize


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def libsize_key(key):
    return f"{key}_{Constants.TENSOR_NAME_LIBSIZE}"


@pytest.fixture
def scale_library_size(key):
    return ScaleLibrarySize(key=key)


def test_init(scale_library_size, key):
    assert isinstance(scale_library_size, ScaleLibrarySize)
    assert scale_library_size.key == key
    assert scale_library_size.show_warning


def test_call_dense(scale_library_size, key, libsize_key):
    inputs = {
        key: tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]]),
        libsize_key: tf.convert_to_tensor([50.0], dtype=tf.float32),
    }
    outputs = scale_library_size(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]]),
        libsize_key: tf.convert_to_tensor([50.0], dtype=tf.float32),
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])


def test_call_sparse(scale_library_size, key, libsize_key):
    dense_tensor = tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {
        key: sparse_tensor,
        libsize_key: tf.convert_to_tensor([50.0], dtype=tf.float32),
    }
    outputs = scale_library_size(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]])),
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


def test_call_dense_no_libsize(capsys, scale_library_size, key, libsize_key):
    inputs = {
        key: tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]]),
    }
    outputs = scale_library_size(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]]),
        libsize_key: tf.convert_to_tensor([1.0], dtype=tf.float32),
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert libsize_key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
    tf.debugging.assert_equal(outputs[libsize_key], expected_outputs[libsize_key])


def test_call_sparse_no_libsize(scale_library_size, key, libsize_key):
    dense_tensor = tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = scale_library_size(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])),
        libsize_key: tf.convert_to_tensor([1.0], dtype=tf.float32),
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
