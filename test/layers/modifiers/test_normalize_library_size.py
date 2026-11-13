import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers import NormalizeLibrarySize
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in normalize_library_size."""
    return mock_tensor_utils("normalize_library_size")

LIB = X.sum(-1, keepdims=True)


def test_dense_rows_sum_to_one_and_libsize_stored():
    out = NormalizeLibrarySize("rna")({"rna": dense()})
    np.testing.assert_allclose(to_np(out["rna"]), X / LIB, rtol=1e-6)
    np.testing.assert_allclose(to_np(out["rna"]).sum(-1), 1.0, rtol=1e-6)
    lib = out[f"rna_{Constants.TENSOR_NAME_LIBSIZE}"]
    assert tuple(lib.shape) == (2, 1)
    np.testing.assert_allclose(lib.numpy(), LIB)


def test_tuple_key():
    out = NormalizeLibrarySize(("rna", "x"))({("rna", "x"): dense()})
    assert ("rna", Constants.TENSOR_NAME_LIBSIZE) in out


def test_sparse():
    out = NormalizeLibrarySize("rna")({"rna": sparse()})
    assert isinstance(out["rna"], tf.SparseTensor)
    np.testing.assert_allclose(to_np(out["rna"]), X / LIB, rtol=1e-6)


def test_batch_size_one():
    out = NormalizeLibrarySize("rna")({"rna": tf.constant([[1.0, 3.0]])})
    np.testing.assert_allclose(out["rna"].numpy(), [[0.25, 0.75]])


def test_non_iterable_key_raises():
    with pytest.raises(NotImplementedError):
        NormalizeLibrarySize(1)({1: dense()})


def test_zero_library_size_is_finite():
    # regression: all-zero row used to give NaN (0/0); now divide_no_nan keeps it 0
    out = NormalizeLibrarySize("rna")({"rna": tf.constant([[0.0, 0.0], [1.0, 1.0]])})
    assert np.isfinite(out["rna"].numpy()).all()
    np.testing.assert_array_equal(out["rna"].numpy(), [[0.0, 0.0], [0.5, 0.5]])


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    NormalizeLibrarySize("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)
