import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from cavachon.utils import TensorUtils


class TestIsSparseTensor:
    def test_sparse_tensor(self):
        st = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[2, 2])
        assert TensorUtils.is_sparse_tensor(st)

    def test_dense_tensor(self):
        assert not TensorUtils.is_sparse_tensor(tf.zeros((2, 2)))

    def test_keras_inputs(self):
        assert TensorUtils.is_sparse_tensor(tf.keras.Input((3,), sparse=True))
        assert not TensorUtils.is_sparse_tensor(tf.keras.Input((3,)))

    @pytest.mark.parametrize("obj", [None, 1, np.zeros(2), sp.csr_matrix((2, 2))])
    def test_non_tensor(self, obj):
        assert not TensorUtils.is_sparse_tensor(obj)


class TestMaxNNeurons:
    def test_basic(self):
        layers = [
            tf.keras.layers.Dense(4),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(8),
        ]
        assert TensorUtils.max_n_neurons(layers) == 16

    def test_no_dense(self):
        assert TensorUtils.max_n_neurons([tf.keras.layers.Dropout(0.1)]) == 0
        assert TensorUtils.max_n_neurons([]) == 0

    def test_sequential_layers(self):
        seq = TensorUtils.create_backbone_layers(n_layers=3, base_n_neurons=4, max_n_neurons=10)
        assert TensorUtils.max_n_neurons(seq.layers) == 10


class TestRemoveNanGradients:
    def test_replaces_nan_inf_and_clips(self):
        g = [tf.constant([np.nan, np.inf, -np.inf, 20.0, -20.0, 3.0])]
        out = TensorUtils.remove_nan_gradients(g)
        np.testing.assert_allclose(out[0].numpy(), [0, 0, 0, 10, -10, 3])

    def test_none_kept(self):
        out = TensorUtils.remove_nan_gradients([None, tf.constant([1.0])])
        assert out[0] is None
        np.testing.assert_allclose(out[1].numpy(), [1.0])

    def test_custom_clip(self):
        out = TensorUtils.remove_nan_gradients([tf.constant([[5.0, -5.0, 0.5]])], clip_value=1)
        np.testing.assert_allclose(out[0].numpy(), [[1.0, -1.0, 0.5]])

    def test_empty(self):
        assert TensorUtils.remove_nan_gradients([]) == []

    def test_works_inside_tf_function(self):
        @tf.function
        def f(x):
            return TensorUtils.remove_nan_gradients([x])[0]

        np.testing.assert_allclose(f(tf.constant([np.nan, 11.0])).numpy(), [0.0, 10.0])

    @pytest.mark.xfail(
        strict=True,
        reason="BUG (minor): input list is mutated in place; the caller's gradient "
        "list is overwritten",
    )
    def test_does_not_mutate_input(self):
        original = tf.constant([np.nan])
        g = [original]
        TensorUtils.remove_nan_gradients(g)
        assert g[0] is original


class TestCreateBackboneLayers:
    def test_defaults(self):
        seq = TensorUtils.create_backbone_layers()
        assert isinstance(seq, tf.keras.Sequential)
        assert seq.name == "backbone_network"
        dense = [l for l in seq.layers if isinstance(l, tf.keras.layers.Dense)]
        assert [d.units for d in dense] == [32, 64]
        assert all(d.get_config()["activation"] in ("swish", "silu") for d in dense)

    def test_alternating_dense_norm(self):
        seq = TensorUtils.create_backbone_layers(n_layers=3, base_n_neurons=2, max_n_neurons=100)
        kinds = [type(l).__name__ for l in seq.layers]
        assert kinds == ["Dense", "LayerNormalization"] * 3
        assert [l.units for l in seq.layers[::2]] == [2, 4, 8]

    def test_max_neurons_caps(self):
        seq = TensorUtils.create_backbone_layers(n_layers=4, base_n_neurons=8, max_n_neurons=20, rate=3)
        assert [l.units for l in seq.layers[::2]] == [8, 20, 20, 20]

    def test_layer_names_unique_and_prefixed(self):
        seq = TensorUtils.create_backbone_layers(n_layers=2, name="enc")
        names = [l.name for l in seq.layers]
        assert len(set(names)) == len(names)
        assert all(n.startswith("enc_") for n in names)

    def test_zero_layers(self):
        seq = TensorUtils.create_backbone_layers(n_layers=0)
        assert seq.layers == []

    def test_forward_shape(self):
        seq = TensorUtils.create_backbone_layers(n_layers=2, base_n_neurons=4, max_n_neurons=8)
        assert seq(tf.zeros((3, 5))).shape == (3, 8)

    def test_reverse_last_layer_units(self):
        seq = TensorUtils.create_backbone_layers(
            n_layers=3, base_n_neurons=2, max_n_neurons=100, reverse=True
        )
        dense_units = [l.units for l in seq.layers if isinstance(l, tf.keras.layers.Dense)]
        assert dense_units == [8, 4, 2]
        assert seq(tf.zeros((1, 3))).shape == (1, 2)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: reverse=True reverses the flat [Dense, Norm, ...] list, so "
        "LayerNormalization precedes each Dense (net starts with a norm on the "
        "raw input and ends with a bare Dense) instead of Dense->Norm pairs",
    )
    def test_reverse_keeps_dense_norm_pairs(self):
        seq = TensorUtils.create_backbone_layers(n_layers=2, reverse=True)
        kinds = [type(l).__name__ for l in seq.layers]
        assert kinds == ["Dense", "LayerNormalization"] * 2


class TestCreateOneHotEncodedTensor:
    def test_fit_new_encoder(self):
        s = pd.Series(["b", "a", "b", "c"])
        t, enc = TensorUtils.create_one_hot_encoded_tensor(s)
        assert isinstance(enc, LabelEncoder)
        assert list(enc.classes_) == ["a", "b", "c"]
        assert t.dtype == tf.float32
        np.testing.assert_array_equal(
            t.numpy(), [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def test_reuse_encoder_subset_of_classes(self):
        enc = LabelEncoder().fit(["a", "b", "c"])
        t, enc2 = TensorUtils.create_one_hot_encoded_tensor(pd.Series(["c"]), enc)
        assert enc2 is enc
        np.testing.assert_array_equal(t.numpy(), [[0, 0, 1]])

    def test_unseen_label_with_encoder_raises(self):
        enc = LabelEncoder().fit(["a"])
        with pytest.raises(ValueError):
            TensorUtils.create_one_hot_encoded_tensor(pd.Series(["z"]), enc)


class TestCreateTensorFromDf:
    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "A": [0.0] * 6 + [1.0] * 6,
                "B": np.arange(12, dtype=float),
                "C": ["x", "y"] * 6,
            }
        )

    def test_categorical_and_continuous(self, df):
        t, enc = TensorUtils.create_tensor_from_df(df, ["A", "B"], dict())
        assert t.shape == (12, 3)
        assert isinstance(enc["A"], LabelEncoder)
        assert enc["B"] is None
        np.testing.assert_array_equal(t.numpy()[:, 2], np.arange(12))
        np.testing.assert_array_equal(t.numpy()[:6, :2], [[1, 0]] * 6)

    def test_string_column(self, df):
        t, enc = TensorUtils.create_tensor_from_df(df, ["C"], dict())
        assert t.shape == (12, 2)
        assert list(enc["C"].classes_) == ["x", "y"]

    def test_missing_columns_skipped_with_warning(self, df):
        with pytest.warns(RuntimeWarning, match="'nope'"):
            t, enc = TensorUtils.create_tensor_from_df(df, ["nope", "B"], dict())
        assert t.shape == (12, 1)
        assert "nope" not in enc

    def test_all_columns_missing_gives_zero_vector_with_warning(self, df):
        with pytest.warns(RuntimeWarning, match="'x1'"):
            t, enc = TensorUtils.create_tensor_from_df(df, ["x1"], dict())
        assert t.shape == (12, 1)
        assert enc == {}

    def test_present_columns_do_not_warn(self, df):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            TensorUtils.create_tensor_from_df(df, ["A", "B", "C"], dict())

    def test_no_columns_gives_zero_vector(self, df):
        t, enc = TensorUtils.create_tensor_from_df(df, [], dict())
        assert t.shape == (12, 1)
        assert float(tf.reduce_sum(tf.abs(t))) == 0.0
        assert enc == {}

    def test_reuses_provided_encoder(self, df):
        enc = LabelEncoder().fit(["x", "y", "z"])
        t, out = TensorUtils.create_tensor_from_df(df, ["C"], {"C": enc})
        assert out["C"] is enc
        assert t.shape == (12, 3)

    def test_provided_dict_is_updated_in_place(self, df):
        d = {}
        _, out = TensorUtils.create_tensor_from_df(df, ["A"], d)
        assert out is d and "A" in d

    def test_encoder_dict_default_is_none(self):
        default = inspect.signature(TensorUtils.create_tensor_from_df).parameters[
            "encoder_dict"
        ].default
        assert default is None

    def test_default_encoder_dict_not_shared_between_calls(self):
        # regression: mutable default dict() leaked encoders across calls
        df1 = pd.DataFrame({"batch": ["a", "b"] * 10})
        df2 = pd.DataFrame({"batch": ["c", "d", "e"] * 10})
        _, enc1 = TensorUtils.create_tensor_from_df(df1, ["batch"])
        _, enc2 = TensorUtils.create_tensor_from_df(df2, ["batch"])
        assert enc1 is not enc2
        assert list(enc1["batch"].classes_) == ["a", "b"]
        assert list(enc2["batch"].classes_) == ["c", "d", "e"]

    def test_existing_encoder_forces_one_hot_despite_heuristic(self):
        # 6 unique of 6 obs would be 'continuous' by the heuristic; a stored
        # LabelEncoder (fitted on the full data) must win so subsets match.
        full = pd.DataFrame({"b": np.repeat(np.arange(6.0), 10)})
        _, enc = TensorUtils.create_tensor_from_df(full, ["b"], dict())
        assert isinstance(enc["b"], LabelEncoder)
        subset = pd.DataFrame({"b": np.arange(6.0)})
        t, out = TensorUtils.create_tensor_from_df(subset, ["b"], enc)
        assert out["b"] is enc["b"]
        assert t.shape == (6, 6)
        np.testing.assert_array_equal(t.numpy(), np.eye(6))

    def test_existing_none_forces_continuous_despite_heuristic(self):
        # 1 unique value would be 'categorical' by the heuristic; a stored None
        # (continuous on the full data) must win.
        subset = pd.DataFrame({"b": [3.0] * 10})
        enc = {"b": None}
        t, out = TensorUtils.create_tensor_from_df(subset, ["b"], enc)
        assert out == {"b": None}
        assert t.shape == (10, 1)
        np.testing.assert_array_equal(t.numpy()[:, 0], [3.0] * 10)

    def test_heuristic_used_only_for_new_columns(self, df):
        enc = {"A": None}
        t, out = TensorUtils.create_tensor_from_df(df, ["A", "C"], enc)
        assert out["A"] is None  # continuous kept
        assert isinstance(out["C"], LabelEncoder)  # inferred categorical
        assert t.shape == (12, 1 + 2)


class TestSpmatrixToSparseTensor:
    def test_does_not_use_np_mat(self, monkeypatch):
        # regression: np.mat was removed in NumPy 2.0
        monkeypatch.delattr(np, "mat", raising=False)
        monkeypatch.delattr(np, "matrix", raising=False)
        st = TensorUtils.spmatrix_to_sparse_tensor(sp.csr_matrix(np.eye(3)))
        assert st.indices.dtype == tf.int64
        np.testing.assert_array_equal(tf.sparse.to_dense(st).numpy(), np.eye(3))

    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
    def test_roundtrip(self, fmt):
        dense = np.array([[0, 2, 0], [1, 0, 3]], dtype=np.int64)
        m = getattr(sp, f"{fmt}_matrix")(dense)
        st = TensorUtils.spmatrix_to_sparse_tensor(m)
        assert isinstance(st, tf.SparseTensor)
        assert st.dtype == tf.float32
        assert tuple(st.dense_shape.numpy()) == (2, 3)
        np.testing.assert_array_equal(tf.sparse.to_dense(st).numpy(), dense)

    def test_indices_are_ordered(self):
        m = sp.csc_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float))
        st = TensorUtils.spmatrix_to_sparse_tensor(m)
        idx = st.indices.numpy().tolist()
        assert idx == sorted(idx)

    def test_empty_matrix(self):
        st = TensorUtils.spmatrix_to_sparse_tensor(sp.csr_matrix((3, 4)))
        assert tuple(st.dense_shape.numpy()) == (3, 4)
        assert st.values.shape[0] == 0

    def test_sparse_array_input(self):
        st = TensorUtils.spmatrix_to_sparse_tensor(sp.csr_array(np.eye(2)))
        np.testing.assert_array_equal(tf.sparse.to_dense(st).numpy(), np.eye(2))

    def test_dense_ndarray_input(self):
        dense = np.array([[0, 2, 0], [1, 0, 3]], dtype=float)
        st = TensorUtils.spmatrix_to_sparse_tensor(dense)
        np.testing.assert_array_equal(tf.sparse.to_dense(st).numpy(), dense)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: non-canonical sparse input with duplicate entries is not summed",
    )
    def test_duplicate_entries_summed(self):
        m = sp.coo_matrix(([1.0, 2.0], ([0, 0], [0, 0])), shape=(1, 2))
        st = TensorUtils.spmatrix_to_sparse_tensor(m)
        np.testing.assert_array_equal(tf.sparse.to_dense(st).numpy(), [[3.0, 0.0]])


class TestSplit:
    def test_uneven(self):
        parts = TensorUtils.split(tf.range(10), batch_size=4)
        assert [int(p.shape[0]) for p in parts] == [4, 4, 2]
        np.testing.assert_array_equal(tf.concat(parts, 0).numpy(), np.arange(10))

    def test_smaller_than_batch(self):
        parts = TensorUtils.split(tf.zeros((3, 2)), batch_size=128)
        assert [int(p.shape[0]) for p in parts] == [3]

    def test_preserves_trailing_dims(self):
        parts = TensorUtils.split(tf.zeros((5, 2, 3)), batch_size=2)
        assert all(p.shape[1:] == (2, 3) for p in parts)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: when n_obs is a multiple of batch_size an extra empty "
        "(0-row) chunk is appended",
    )
    def test_even_split_has_no_empty_chunk(self):
        parts = TensorUtils.split(tf.range(8), batch_size=4)
        assert [int(p.shape[0]) for p in parts] == [4, 4]
