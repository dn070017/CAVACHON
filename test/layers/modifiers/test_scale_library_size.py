import warnings

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers import NormalizeLibrarySize, ScaleLibrarySize
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in scale_library_size."""
    return mock_tensor_utils("scale_library_size")

KEY = f"rna_{Constants.TENSOR_NAME_LIBSIZE}"


def test_inverse_of_normalize():
    out = NormalizeLibrarySize("rna")({"rna": dense()})
    out = ScaleLibrarySize("rna")(out)
    np.testing.assert_allclose(to_np(out["rna"]), X, rtol=1e-6)


def test_sparse_round_trip():
    out = NormalizeLibrarySize("rna")({"rna": sparse()})
    out = ScaleLibrarySize("rna")(out)
    assert isinstance(out["rna"], tf.SparseTensor)
    np.testing.assert_allclose(to_np(out["rna"]), X, rtol=1e-6)


def test_missing_libsize_warns_once_and_is_identity():
    layer = ScaleLibrarySize("rna")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = layer({"rna": dense()})
        layer({"rna": dense()})
    assert sum(issubclass(x.category, RuntimeWarning) for x in w) == 1
    np.testing.assert_allclose(out["rna"].numpy(), X)
    np.testing.assert_allclose(out[KEY].numpy(), np.ones((2, 1)))


def test_tuple_key():
    inputs = {("rna", "x"): dense(), ("rna", Constants.TENSOR_NAME_LIBSIZE): tf.constant([[2.0], [3.0]])}
    out = ScaleLibrarySize(("rna", "x"))(inputs)
    np.testing.assert_allclose(out[("rna", "x")].numpy(), X * np.array([[2.0], [3.0]]))


def test_non_iterable_key_raises():
    with pytest.raises(NotImplementedError):
        ScaleLibrarySize(1)({1: dense()})


@pytest.mark.xfail(
    strict=True,
    reason="BUG: scale_library_size.py:78 builds default libsize with tf.ones((tensor.shape[0], 1)); "
    "fails when batch dim is unknown (tf.function with None batch)",
)
def test_missing_libsize_with_unknown_batch_dim():
    layer = ScaleLibrarySize("rna")

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def f(t):
        return layer({"rna": t})["rna"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_allclose(f(dense()).numpy(), X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    ScaleLibrarySize("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)
