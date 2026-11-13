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


@pytest.mark.xfail(
    strict=True,
    reason="BUG: to_sparse.py:51-53 only converts KerasTensor/SparseTensor; a dense tf.Tensor is "
    "returned unchanged (and a SparseTensor branch is dead code)",
)
def test_dense_eager_tensor_is_converted():
    out = ToSparse("a")({"a": dense()})["a"]
    assert isinstance(out, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(out), X)



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
