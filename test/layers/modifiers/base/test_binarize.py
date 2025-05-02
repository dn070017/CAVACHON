import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.binarize import Binarize


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def threshold():
    return 0.5


@pytest.fixture
def binarize(key, threshold):
    return Binarize(key=key, threshold=threshold)


def test_init(binarize, key, threshold):
    assert isinstance(binarize, Binarize)
    assert binarize.key == key
    assert binarize.threshold == threshold


def test_call_dense(binarize, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])}
    outputs = binarize(inputs)
    expected_outputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.0, 1.0])}
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])


def test_call_sparse(binarize, key):
    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = binarize(inputs)
    expected_outputs = {
        key: tf.sparse.from_dense(tf.convert_to_tensor([0.0, 1.0, 1.0, 0.0, 1.0]))
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    assert isinstance(outputs[key], tf.SparseTensor)
    tf.debugging.assert_equal(outputs[key].indices, expected_outputs[key].indices)
    tf.debugging.assert_equal(outputs[key].values, expected_outputs[key].values)
    tf.debugging.assert_equal(
        outputs[key].dense_shape, expected_outputs[key].dense_shape
    )


def test_invalid_threshold():
    binarize = Binarize(key="test_key", threshold=1.5)
    assert binarize.threshold == 1.0


def test_binarize_get_config(binarize, key, threshold):
    config = binarize.get_config()
    new_binarize = Binarize.from_config(config)
    assert new_binarize.key == key
    assert new_binarize.threshold == threshold
