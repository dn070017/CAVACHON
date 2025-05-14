from contextlib import nullcontext

import pytest
import tensorflow as tf

from cavachon.layers.base.verify_io_layer import VerifyIOLayer


def test_verify_tensor_like():
    layer = VerifyIOLayer(expected_input_keys=None, expected_output_keys=None)
    value = tf.constant(1.0)
    layer.verify_tensor_like(value, "test_value")

    with pytest.raises(TypeError):
        layer.verify_tensor_like("not a tensor", "test_value")


def test_verify_io_keys():
    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=None
    )
    input_dict = {"key1": tf.constant(1.0), "key2": tf.constant(2.0)}
    layer.verify_io_keys(input_dict, ["key1", "key2"], "test_input")

    with pytest.raises(KeyError):
        layer.verify_io_keys({"key1": tf.constant(1.0)}, ["key1", "key2"], "test_input")

    with pytest.raises(TypeError):
        layer.verify_io_keys("not a dict", ["key1", "key2"], "test_input")


def test_verify_inputs():
    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=None
    )
    layer.verify_inputs({"key1": tf.constant(1.0), "key2": tf.constant(2.0)})

    with pytest.raises(TypeError):
        layer.verify_inputs(tf.constant(1.0))

    with pytest.raises(KeyError):
        layer.verify_inputs({"key1": tf.constant(1.0)})

    layer = VerifyIOLayer(expected_input_keys=None, expected_output_keys=None)
    with pytest.raises(TypeError):
        layer.verify_inputs({"key1": tf.constant(1.0)})


def test_verify_outputs():
    layer = VerifyIOLayer(
        expected_input_keys=None, expected_output_keys=["key1", "key2"]
    )
    layer.verify_outputs({"key1": tf.constant(1.0), "key2": tf.constant(2.0)})

    with pytest.raises(TypeError):
        layer.verify_outputs(tf.constant(1.0))

    with pytest.raises(KeyError):
        layer.verify_outputs({"key1": tf.constant(1.0)})

    layer = VerifyIOLayer(expected_input_keys=None, expected_output_keys=None)
    with pytest.raises(TypeError):
        layer.verify_outputs({"key1": tf.constant(1.0)})


def test_binarize_get_config():
    layer = VerifyIOLayer(
        expected_input_keys=None, expected_output_keys=["key1", "key2"]
    )
    config = layer.get_config()
    new_layer = VerifyIOLayer.from_config(config)
    assert new_layer.expected_input_keys is None
    assert new_layer.expected_output_keys == ["key1", "key2"]

    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=None
    )
    config = layer.get_config()
    new_layer = VerifyIOLayer.from_config(config)
    assert new_layer.expected_input_keys == ["key1", "key2"]
    assert new_layer.expected_output_keys is None


def test_check_n_expected_keys_correct():
    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=["key3"]
    )
    with nullcontext():
        layer.check_n_expected_keys(n_expected_inputs=2, n_expected_outputs=1)


def test_check_n_expected_keys_incorrect_inputs():
    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=["key3"]
    )
    with pytest.raises(ValueError):
        layer.check_n_expected_keys(n_expected_inputs=1, n_expected_outputs=1)


def test_check_n_expected_keys_incorrect_outputs():
    layer = VerifyIOLayer(
        expected_input_keys=["key1", "key2"], expected_output_keys=["key3"]
    )
    with pytest.raises(ValueError):
        layer.check_n_expected_keys(n_expected_inputs=2, n_expected_outputs=2)


def test_get_tensor_with_single_key():
    layer = VerifyIOLayer(expected_input_keys=["key1"])
    input_dict = {"key1": tf.constant(1.0), "key2": tf.constant(2.0)}
    tensor = layer.get_tensor_with_single_key(input_dict)
    tf.debugging.assert_equal(tensor, input_dict["key1"])


def test_get_tensor_with_single_key_not_dict():
    layer = VerifyIOLayer()
    inputs = tf.constant(1.0)
    tensor = layer.get_tensor_with_single_key(inputs)
    tf.debugging.assert_equal(tensor, inputs)


def test_get_tensor_with_single_key_multiple_keys():
    layer = VerifyIOLayer(expected_input_keys=["key1", "key2"])
    input_dict = {"key1": tf.constant(1.0), "key2": tf.constant(2.0)}
    with pytest.raises(NotImplementedError):
        layer.get_tensor_with_single_key(input_dict)


def test_set_tensor_with_single_key():
    layer = VerifyIOLayer(expected_output_keys=["key1"])
    input_dict = {"key2": tf.constant(2.0)}
    output_dict = layer.set_tensor_with_single_key(input_dict, tf.constant(1.0))
    tf.debugging.assert_equal(output_dict["key1"], tf.constant(1.0))
    tf.debugging.assert_equal(output_dict["key2"], tf.constant(2.0))
    assert "key1" not in input_dict


def test_set_tensor_with_single_key_not_dict():
    layer = VerifyIOLayer()
    input_dict = {"key2": tf.constant(2.0)}
    output = layer.set_tensor_with_single_key(input_dict, tf.constant(1.0))
    tf.debugging.assert_equal(output, tf.constant(1.0))


def test_set_tensor_with_single_key_multiple_keys():
    layer = VerifyIOLayer(expected_output_keys=["key1", "key2"])
    input_dict = {"key2": tf.constant(2.0)}
    with pytest.raises(NotImplementedError):
        layer.set_tensor_with_single_key(input_dict, tf.constant(1.0))
