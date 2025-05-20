import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.binarize import Binarize


@pytest.fixture
def threshold():
    return 0.5


@pytest.fixture
def binarize(threshold):
    return Binarize(
        threshold=threshold,
    )


def test_call(binarize):
    inputs = tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])
    outputs = binarize(inputs)
    expected_outputs = tf.convert_to_tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    tf.debugging.assert_equal(expected_outputs, outputs)


def test_call_zero_threshold():
    inputs = tf.convert_to_tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    outputs = Binarize(threshold=0.0)(inputs)
    expected_outputs = tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_call_one_threshold():
    inputs = tf.convert_to_tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    outputs = Binarize(threshold=1.0)(inputs)
    expected_outputs = tf.convert_to_tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_call_large_tensor(binarize):
    inputs = tf.random.uniform(shape=(1000,), minval=0.0, maxval=1.0)
    outputs = binarize(inputs)
    expected_outputs = tf.where(
        inputs >= binarize.threshold,
        tf.ones_like(inputs),
        tf.zeros_like(inputs),
    )
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_call_dense_all_below_threshold(binarize):
    inputs = tf.convert_to_tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    outputs = binarize(inputs)
    expected_outputs = tf.convert_to_tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_invalid_threshold():
    with pytest.warns():
        binarize = Binarize(threshold=1.5)
        assert binarize.threshold == 1.0

    with pytest.warns():
        binarize = Binarize(threshold=-1.5)
        assert binarize.threshold == 0.0


def test_call_dense_all_above_threshold(binarize):
    inputs = tf.convert_to_tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    outputs = binarize(inputs)
    expected_outputs = tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_binarize_get_config(binarize, threshold):
    config = binarize.get_config()
    new_binarize = Binarize.from_config(config)
    assert new_binarize.threshold == threshold
