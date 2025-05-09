import pytest
import tensorflow as tf

from cavachon.layers.base.verify_io_layer import VerifyIOLayer


def test_verify_tensor_like():
    layer = VerifyIOLayer(input_keys=None, output_keys=None)
    value = tf.constant(1.0)
    layer.verify_tensor_like(value, "test_value")

    with pytest.raises(TypeError):
        layer.verify_tensor_like("not a tensor", "test_value")


def test_verify_io_keys():
    layer = VerifyIOLayer(input_keys=["key1", "key2"], output_keys=None)
    input_dict = {"key1": tf.constant(1.0), "key2": tf.constant(2.0)}
    layer.verify_io_keys(input_dict, ["key1", "key2"], "test_input")

    with pytest.raises(KeyError):
        layer.verify_io_keys({"key1": tf.constant(1.0)}, ["key1", "key2"], "test_input")

    with pytest.raises(TypeError):
        layer.verify_io_keys("not a dict", ["key1", "key2"], "test_input")


def test_verify_inputs():
    layer = VerifyIOLayer(input_keys=["key1", "key2"], output_keys=None)
    layer.verify_inputs({"key1": tf.constant(1.0), "key2": tf.constant(2.0)})

    with pytest.raises(TypeError):
        layer.verify_inputs(tf.constant(1.0))

    with pytest.raises(KeyError):
        layer.verify_inputs({"key1": tf.constant(1.0)})

    layer = VerifyIOLayer(input_keys=None, output_keys=None)
    with pytest.raises(TypeError):
        layer.verify_inputs({"key1": tf.constant(1.0)})


def test_verify_outputs():
    layer = VerifyIOLayer(input_keys=None, output_keys=["key1", "key2"])
    layer.verify_outputs({"key1": tf.constant(1.0), "key2": tf.constant(2.0)})

    with pytest.raises(TypeError):
        layer.verify_outputs(tf.constant(1.0))

    with pytest.raises(KeyError):
        layer.verify_outputs({"key1": tf.constant(1.0)})

    layer = VerifyIOLayer(input_keys=None, output_keys=None)
    with pytest.raises(TypeError):
        layer.verify_outputs({"key1": tf.constant(1.0)})


def test_binarize_get_config():
    layer = VerifyIOLayer(input_keys=None, output_keys=["key1", "key2"])
    config = layer.get_config()
    new_layer = VerifyIOLayer.from_config(config)
    assert new_layer.input_keys is None
    assert new_layer.output_keys == ["key1", "key2"]

    layer = VerifyIOLayer(input_keys=["key1", "key2"], output_keys=None)
    config = layer.get_config()
    new_layer = VerifyIOLayer.from_config(config)
    assert new_layer.input_keys == ["key1", "key2"]
    assert new_layer.output_keys is None
