import math

import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.exp_transform import ExpTransform


@pytest.fixture()
def exp_transform():
    return ExpTransform()


@pytest.fixture()
def exp_transform_base2():
    return ExpTransform(base=2)


def test_init_default(exp_transform):
    assert isinstance(exp_transform, ExpTransform)
    assert exp_transform.base == math.e


def test_init_base2(exp_transform_base2):
    assert isinstance(exp_transform_base2, ExpTransform)
    assert exp_transform_base2.base == 2


def test_call(exp_transform):
    inputs = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    outputs = exp_transform(inputs)
    expected_outputs = tf.convert_to_tensor(
        [1.0, np.exp(1.0), np.exp(2.0), np.exp(3.0)]
    )
    tf.debugging.assert_near(outputs, expected_outputs, rtol=1e-6)


def test_call_base2(exp_transform_base2):
    inputs = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    outputs = exp_transform_base2(inputs)
    expected_outputs = tf.convert_to_tensor([1.0, 2.0, 4.0, 8.0])
    tf.debugging.assert_near(outputs, expected_outputs, rtol=1e-6)


def test_exp_transform_get_config(exp_transform):
    config = exp_transform.get_config()
    new_exp_transform = ExpTransform.from_config(config)
    assert new_exp_transform.base == math.e
