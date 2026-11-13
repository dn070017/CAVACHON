import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers import ExpTransform, LogTransform
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in log_transform."""
    return mock_tensor_utils("log_transform")


def test_default_pseudocount_is_log1p():
    out = LogTransform("a")({"a": dense()})["a"]
    np.testing.assert_allclose(to_np(out), np.log1p(X), rtol=1e-6)


@pytest.mark.parametrize("pc", [0.5, 2.0])
def test_custom_pseudocount(pc):
    out = LogTransform("a", pseudocount=pc)({"a": dense()})["a"]
    np.testing.assert_allclose(to_np(out), np.log(X + pc), rtol=1e-6)


def test_sparse_preserves_sparsity_pattern():
    out = LogTransform("a")({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)
    assert out.indices.shape[0] == np.count_nonzero(X)
    np.testing.assert_allclose(to_np(out), np.log1p(X), rtol=1e-6)


def test_zero_pseudocount_gives_neg_inf_on_zero():
    out = LogTransform("a", pseudocount=0.0)({"a": tf.zeros((1, 2))})["a"]
    assert np.isneginf(out.numpy()).all()


def test_inverse_of_exp_minus_pseudocount():
    out = LogTransform("a")({"a": dense()})
    back = ExpTransform("a")(out)["a"]
    np.testing.assert_allclose(to_np(back) - 1.0, X, atol=1e-6)


def test_get_config_round_trip():
    clone = LogTransform.from_config(LogTransform("a", pseudocount=3.0).get_config())
    assert clone.key == "a" and clone.pseudocount == 3.0


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    LogTransform("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)


def test_sparse_path_taken_when_tensor_utils_says_sparse(tensor_utils):
    tensor_utils.is_sparse_tensor.side_effect = None
    tensor_utils.is_sparse_tensor.return_value = True
    out = LogTransform("a")({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)
    tensor_utils.is_sparse_tensor.assert_called_once()
