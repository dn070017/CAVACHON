import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.scale_constant_factor import ScaleConstantFactor


@pytest.fixture
def scale_constant_factor():
    return ScaleConstantFactor()


@pytest.fixture
def scale_constant_factor_10():
    return ScaleConstantFactor(scaling_factor=10)


def test_init(scale_constant_factor):
    assert isinstance(scale_constant_factor, ScaleConstantFactor)
    assert scale_constant_factor.scaling_factor == 1e7


def test_call_dense(scale_constant_factor_10):
    inputs = tf.convert_to_tensor([[0.1, 0.1, 0.2, 0.4, 0.2]])
    outputs = scale_constant_factor_10(inputs)
    expected_outputs = tf.convert_to_tensor([[1.0, 1.0, 2.0, 4.0, 2.0]])
    tf.debugging.assert_equal(outputs, expected_outputs)


def test_scale_constant_factor_get_config(scale_constant_factor):
    config = scale_constant_factor.get_config()
    new_scale_constant_factor = ScaleConstantFactor.from_config(config)
    assert new_scale_constant_factor.scaling_factor == 1e7


def test_invalid_scaling_factor(scale_constant_factor):
    with pytest.warns():
        scale_constant_factor = ScaleConstantFactor(scaling_factor=-1)
        assert scale_constant_factor.scaling_factor == 1e7
