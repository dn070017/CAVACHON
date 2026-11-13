import pytest
import numpy as np
import tensorflow as tf

from cavachon.layers.modifiers import ToDense
from ._helpers import X, dense, sparse


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in to_dense."""
    return mock_tensor_utils("to_dense")


def test_sparse_to_dense():
    out = ToDense("a")({"a": sparse()})["a"]
    assert not isinstance(out, tf.SparseTensor)
    np.testing.assert_array_equal(out.numpy(), X)


def test_dense_passthrough():
    out = ToDense("a")({"a": dense()})["a"]
    np.testing.assert_array_equal(out.numpy(), X)


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    ToDense("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)


def test_conversion_driven_by_tensor_utils(tensor_utils):
    tensor_utils.is_sparse_tensor.side_effect = None
    tensor_utils.is_sparse_tensor.return_value = False
    out = ToDense("a")({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)  # untouched when reported dense
