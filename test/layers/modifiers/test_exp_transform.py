import pytest
import numpy as np
import tensorflow as tf

from cavachon.layers.modifiers import ExpTransform
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in exp_transform."""
    return mock_tensor_utils("exp_transform")


def test_dense():
    out = ExpTransform("a")({"a": dense()})["a"]
    np.testing.assert_allclose(to_np(out), np.exp(X), rtol=1e-6)


def test_sparse_stays_sparse_and_zeros_map_to_one():
    out = ExpTransform("a")({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)
    np.testing.assert_allclose(to_np(out), np.exp(X), rtol=1e-6)
    # exp(0) = 1 so the result is fully populated
    assert out.indices.shape[0] == X.size


def test_large_values_overflow_to_inf():
    out = ExpTransform("a")({"a": tf.constant([[100.0]])})["a"]
    assert np.isinf(out.numpy()).all()


def test_get_config_round_trip():
    assert ExpTransform.from_config(ExpTransform("a").get_config()).key == "a"


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    ExpTransform("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)
