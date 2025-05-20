import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.normalize_library_size import NormalizeLibrarySize


@pytest.fixture
def normalize_library_size():
    return NormalizeLibrarySize()


def test_init(normalize_library_size):
    assert isinstance(normalize_library_size, NormalizeLibrarySize)


def test_call(normalize_library_size):
    inputs = tf.convert_to_tensor([[5.0, 5.0, 10.0, 20.0, 10.0]])
    outputs = normalize_library_size(inputs)
    expected_outputs = tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])

    tf.debugging.assert_equal(outputs, expected_outputs)
