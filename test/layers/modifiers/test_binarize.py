import warnings

import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.modifiers import Binarize
from ._helpers import X, dense, sparse, to_np


@pytest.fixture(autouse=True)
def tensor_utils(mock_tensor_utils):
    """cavachon.utils.TensorUtils mocked at its lookup site in binarize."""
    return mock_tensor_utils("binarize")


def test_default_threshold_one():
    out = Binarize("a")({"a": dense()})["a"]
    np.testing.assert_array_equal(to_np(out), (X >= 1.0).astype(np.float32))


@pytest.mark.parametrize("thr", [0.0, 0.5, 1.0])
def test_threshold_dense(thr):
    out = Binarize("a", threshold=thr)({"a": dense()})["a"]
    np.testing.assert_array_equal(to_np(out), (X >= thr).astype(np.float32))


def test_sparse_input_stays_sparse():
    out = Binarize("a", threshold=0.5)({"a": sparse()})["a"]
    assert isinstance(out, tf.SparseTensor)
    np.testing.assert_array_equal(to_np(out), (X >= 0.5).astype(np.float32))


def test_other_keys_untouched():
    other = tf.constant([1.0])
    out = Binarize("a")({"a": dense(), "b": other})
    np.testing.assert_array_equal(out["b"].numpy(), [1.0])


def test_negative_and_zero_values_become_zero():
    t = tf.constant([[-3.0, 0.0, 0.99, 1.0, 5.0]])
    out = Binarize("a")({"a": t})["a"]
    np.testing.assert_array_equal(out.numpy(), [[0, 0, 0, 1, 1]])


def test_tuple_key():
    out = Binarize(("m", "x"), threshold=0.5)({("m", "x"): dense()})
    np.testing.assert_array_equal(to_np(out[("m", "x")]), (X >= 0.5).astype(np.float32))


def test_get_config_round_trip():
    layer = Binarize("a", threshold=0.25, name="bin_cfg")
    cfg = layer.get_config()
    assert cfg["key"] == "a" and cfg["threshold"] == 0.25
    clone = Binarize.from_config(cfg)
    assert clone.key == "a" and clone.threshold == 0.25 and clone.name == "bin_cfg"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: binarize.py:42 uses __name___ (3 underscores) -> AttributeError instead of warning",
)
@pytest.mark.parametrize("thr", [2.0, -0.5])
def test_out_of_range_threshold_warns_and_resets(thr):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer = Binarize("a", threshold=thr)
    assert layer.threshold == 1.0
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_sparsity_check_delegated_to_tensor_utils(tensor_utils):
    t = dense()
    Binarize("a")({"a": t})
    tensor_utils.is_sparse_tensor.assert_called()
    assert isinstance(tensor_utils.is_sparse_tensor.call_args.args[0], tf.Tensor)
