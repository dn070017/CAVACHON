import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers import ToSparse
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in to_sparse."""
    return mock_tensor_utils("to_sparse")


def test_sparse_passthrough():
    out = ToSparse("a")({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(out), X)


def test_dense_eager_tensor_is_converted():
    # regression: only KerasTensor/SparseTensor inputs were considered, so a
    # dense eager tf.Tensor was returned unchanged
    out = ToSparse("a")({"a": dense()})["a"]
    assert isinstance(out, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(out), X)


def test_dense_tensor_inside_tf_function_is_converted():
    layer = ToSparse("a")

    @tf.function
    def f(x):
        out = layer({"a": x})["a"]
        assert isinstance(out, tf.SparseTensor)
        return tf.sparse.to_dense(out)

    np.testing.assert_array_equal(f(dense()).numpy(), X)


def test_dense_keras_input_is_converted_in_functional_model():
    inp = tf.keras.Input(shape=(X.shape[1],))
    out = ToSparse("a")({"a": inp})["a"]
    model = tf.keras.Model(inp, out)
    result = model(dense())
    assert isinstance(result, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(result), X)


def test_other_keys_untouched():
    other = dense()
    out = ToSparse("a")({"a": dense(), "b": other})
    assert out["b"] is other


def test_sparse_input_checked_via_tensor_utils(tensor_utils):
    ToSparse("a")({"a": sparse()})
    tensor_utils.is_sparse_tensor.assert_called_once()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.SparseTensor)


def test_sparse_reported_dense_by_tensor_utils_is_returned_as_is(tensor_utils):
    # SparseTensor branch: if TensorUtils says "not sparse" the layer calls
    # tf.sparse.from_dense on a SparseTensor -> error is surfaced, not hidden.
    tensor_utils.is_sparse_tensor.side_effect = None
    tensor_utils.is_sparse_tensor.return_value = False
    with pytest.raises(Exception):
        ToSparse("a")({"a": sparse()})
