import math

import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.log_transform import LogTransform


@pytest.fixture
def pseudocount():
    return 1.0


@pytest.fixture
def log_transform(pseudocount):
    return LogTransform(pseudocount=pseudocount)


@pytest.fixture
def log_transform_base2(pseudocount):
    return LogTransform(base=2.0, pseudocount=pseudocount)


def test_init(log_transform, pseudocount):
    assert isinstance(log_transform, LogTransform)
    assert log_transform.base == math.e
    assert log_transform.pseudocount == pseudocount


def test_call(log_transform):
    inputs = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    outputs = log_transform(inputs)
    expected_outputs = tf.convert_to_tensor(
        [0.0, np.log(2.0), np.log(3.0), np.log(4.0)]
    )
    tf.debugging.assert_near(outputs, expected_outputs, rtol=1e-6)


def test_call_base2(log_transform_base2):
    inputs = tf.convert_to_tensor([0.0, 1.0, 3.0, 7.0])
    outputs = log_transform_base2(inputs)
    expected_outputs = tf.convert_to_tensor([0.0, 1.0, 2.0, 3.0])
    tf.debugging.assert_near(outputs, expected_outputs, rtol=1e-6)


def test_log_transform_get_config(log_transform, pseudocount):
    config = log_transform.get_config()
    new_log_transform = LogTransform.from_config(config)
    assert new_log_transform.base == math.e
    assert new_log_transform.pseudocount == pseudocount


def test_invalid_base():
    with pytest.warns():
        log_transform = LogTransform(base=-1.0)
        assert log_transform.base == math.e


def test_invalid_pseudocount():
    with pytest.warns():
        log_transform = LogTransform(pseudocount=-1.0)
        assert log_transform.pseudocount == 1.0
