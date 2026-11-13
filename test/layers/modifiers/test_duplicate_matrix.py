import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers import DuplicateMatrix
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in duplicate_matrix."""
    return mock_tensor_utils("duplicate_matrix")


def test_str_key_duplicates_dense():
    out = DuplicateMatrix("rna")({"rna": dense()})
    key = f"rna_{Constants.TENSOR_NAME_ORIGIONAL_X}"
    assert key in out
    np.testing.assert_array_equal(to_np(out[key]), X)
    np.testing.assert_array_equal(to_np(out["rna"]), X)


def test_tuple_key_replaces_last_element():
    out = DuplicateMatrix(("rna", "x"))({("rna", "x"): dense()})
    key = ("rna", Constants.TENSOR_NAME_ORIGIONAL_X)
    assert key in out
    np.testing.assert_array_equal(to_np(out[key]), X)


def test_sparse_duplicate_is_sparse():
    out = DuplicateMatrix("rna")({"rna": sparse()})
    dup = out[f"rna_{Constants.TENSOR_NAME_ORIGIONAL_X}"]
    assert isinstance(dup, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(dup), X)


def test_non_iterable_key_raises():
    with pytest.raises(NotImplementedError):
        DuplicateMatrix(3)({3: dense()})


def test_get_config_round_trip():
    cfg = DuplicateMatrix("rna").get_config()
    assert DuplicateMatrix.from_config(cfg).key == "rna"


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    DuplicateMatrix("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)
