import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers.base.binarize import Binarize
from cavachon.layers.modifiers.base.exp_transform import ExpTransform
from cavachon.layers.modifiers.base.to_sparse import ToSparse
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)


@pytest.fixture
def key():
    return "test_key"


@pytest.fixture
def modifier(key):
    modifier = DistributionPresetModifier()
    return modifier


@pytest.fixture
def one_modifier(key):
    modifier = DistributionPresetModifier()
    modifier.modifiers = [Binarize(key=key, threshold=0.5)]
    return modifier


@pytest.fixture
def two_modifiers_binarize_sparse(key):
    modifier = DistributionPresetModifier()
    modifier.modifiers = [Binarize(key=key, threshold=0.5), ToSparse(key=key)]

    return modifier


@pytest.fixture
def two_modifiers_binarize_exp(key):
    modifier = DistributionPresetModifier()
    modifier.modifiers = [Binarize(key=key, threshold=0.5), ExpTransform(key=key)]
    return modifier


def test_init(modifier):
    assert isinstance(modifier, DistributionPresetModifier)
    assert modifier.modality_name == ""
    assert modifier.modality_key == ""
    assert isinstance(modifier.modifiers, list)
    assert len(modifier.modifiers) == 0


def test_call_without_modifier(modifier, key):
    inputs = {key: tf.convert_to_tensor([0.2, 0.7, 0.1, 0.6, 0.3])}
    outputs = modifier(inputs)
    expected_outputs = {key: tf.convert_to_tensor([0.2, 0.7, 0.1, 0.6, 0.3])}
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])


def test_call_with_single_modifier(one_modifier, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])}
    outputs = one_modifier(inputs)
    expected_outputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.0, 1.0])}
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])

    dense_tensor = tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])
    sparse_tensor = tf.sparse.from_dense(dense_tensor)
    inputs = {key: sparse_tensor}
    outputs = one_modifier(inputs)
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


def test_call_with_multiple_modifier_binarize_sparse(
    two_modifiers_binarize_sparse, key
):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])}
    outputs = two_modifiers_binarize_sparse(inputs)
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


def test_call_with_multiple_modifier_binarize_exp(two_modifiers_binarize_exp, key):
    inputs = {key: tf.convert_to_tensor([0.0, 1.0, 1.0, 0.2, 0.7])}
    outputs = two_modifiers_binarize_exp(inputs)
    expected_outputs = {
        key: tf.convert_to_tensor([1.0, np.exp(1.0), np.exp(1.0), 1.0, np.exp(1.0)])
    }
    assert isinstance(outputs, dict)
    assert key in outputs
    tf.debugging.assert_equal(outputs[key], expected_outputs[key])
