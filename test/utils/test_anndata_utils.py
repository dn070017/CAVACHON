import warnings

import muon as mu
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from cavachon.utils import AnnDataUtils

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")


def make_adata(sparse=False, n_obs=4, n_vars=3):
    X = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    obs = pd.DataFrame(
        {"label": list("xxyy")[:n_obs], "num": np.arange(n_obs, dtype=float)},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_type": ["g"] * n_vars}, index=[f"g{j}" for j in range(n_vars)]
    )
    return AnnData(X=sp.csr_matrix(X) if sparse else X, obs=obs, var=var)


def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


class TestReorderOrFilterAdataObs:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_reorder(self, sparse):
        adata = make_adata(sparse)
        out = AnnDataUtils.reorder_or_filter_adata_obs(
            adata, pd.Index(["c3", "c1", "c0", "c2"])
        )
        assert list(out.obs_names) == ["c3", "c1", "c0", "c2"]
        np.testing.assert_array_equal(to_dense(out.X), to_dense(adata.X)[[3, 1, 0, 2]])
        assert sp.issparse(out.X) == sparse
        pd.testing.assert_frame_equal(out.var, adata.var)
        assert out.obs["num"].tolist() == [3.0, 1.0, 0.0, 2.0]

    def test_filter(self):
        adata = make_adata()
        out = AnnDataUtils.reorder_or_filter_adata_obs(adata, pd.Index(["c2"]))
        assert out.shape == (1, 3)
        np.testing.assert_array_equal(out.X, [[6, 7, 8]])

    @pytest.mark.parametrize("sparse", [False, True])
    def test_output_float32(self, sparse):
        adata = make_adata(sparse)
        adata.X = adata.X.astype(np.int64)
        out = AnnDataUtils.reorder_or_filter_adata_obs(adata, adata.obs_names)
        assert out.X.dtype == np.float32
        assert sp.issparse(out.X) == sparse

    def test_does_not_mutate_input(self):
        adata = make_adata()
        before = adata.X.copy()
        names = list(adata.obs_names)
        AnnDataUtils.reorder_or_filter_adata_obs(adata, pd.Index(["c1", "c0"]))
        np.testing.assert_array_equal(adata.X, before)
        assert list(adata.obs_names) == names

    def test_empty_index(self):
        out = AnnDataUtils.reorder_or_filter_adata_obs(
            make_adata(), pd.Index([], dtype=object)
        )
        assert out.shape == (0, 3)

    def test_duplicate_index_entries_duplicate_rows(self):
        out = AnnDataUtils.reorder_or_filter_adata_obs(
            make_adata(), pd.Index(["c0", "c0"])
        )
        assert out.n_obs == 2

    def test_missing_obs_raises_keyerror(self):
        with pytest.raises(KeyError):
            AnnDataUtils.reorder_or_filter_adata_obs(make_adata(), pd.Index(["nope"]))

    def test_non_anndata_warns_and_returns_none(self):
        with pytest.warns(RuntimeWarning, match="not an AnnData"):
            assert (
                AnnDataUtils.reorder_or_filter_adata_obs(np.zeros((2, 2)), pd.Index([]))
                is None
            )

    def test_drops_layers_obsm_uns(self):
        # documents current behaviour: only X/obs/var are carried over
        adata = make_adata()
        adata.layers["counts"] = adata.X.copy()
        adata.obsm["emb"] = np.zeros((4, 2))
        adata.uns["k"] = 1
        out = AnnDataUtils.reorder_or_filter_adata_obs(adata, adata.obs_names)
        assert "counts" not in out.layers
        assert "emb" not in out.obsm
        assert "k" not in out.uns

    def test_no_deprecation_warning(self):
        # regression: AnnData(dtype=...) emitted a FutureWarning (removed API)
        adata = make_adata()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            AnnDataUtils.reorder_or_filter_adata_obs(adata, adata.obs_names)


def make_mdata(sparse=False, rna_extra=None):
    rna = make_adata(sparse)
    rna.obs["batch"] = ["p", "p", "q", "q"]
    atac_X = np.array([[1, 0], [3, 0], [0, 5], [0, 7]], dtype=np.float32)
    atac = AnnData(
        X=sp.csr_matrix(atac_X) if sparse else atac_X,
        obs=pd.DataFrame({"depth": [1.0, 3.0, 5.0, 7.0]}, index=rna.obs_names.copy()),
        var=pd.DataFrame(index=["p0", "p1"]),
    )
    return mu.MuData({"rna": rna, "atac": atac})


class TestMergeMdataOnObsAnnotation:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_group_means(self, sparse):
        mdata = make_mdata(sparse)
        out = AnnDataUtils.merge_mdata_on_obs_annotation(
            mdata, "label", {"rna": [], "atac": []}
        )
        assert isinstance(out, mu.MuData)
        assert set(out.mod) == {"rna", "atac"}
        np.testing.assert_allclose(
            to_dense(out["rna"].X), [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]
        )
        # atac lacks 'label' -> filtered with rna's obs index
        np.testing.assert_allclose(to_dense(out["atac"].X), [[2.0, 0.0], [0.0, 6.0]])
        assert list(out["rna"].obs_names) == ["x", "y"]
        assert list(out["atac"].obs["label"]) == ["x", "y"]
        assert list(out["atac"].var_names) == ["p0", "p1"]
        assert list(out["rna"].var_names) == ["g0", "g1", "g2"]

    def test_output_is_sparse_csr(self):
        out = AnnDataUtils.merge_mdata_on_obs_annotation(
            make_mdata(), "label", {"rna": [], "atac": []}
        )
        assert sp.issparse(out["rna"].X)

    def test_batch_effect_categorical_mode_and_numeric_mean(self):
        mdata = make_mdata()
        out = AnnDataUtils.merge_mdata_on_obs_annotation(
            mdata, "label", {"rna": ["batch", "missing_col"], "atac": ["depth"]}
        )
        assert list(out["rna"].obs["batch"]) == ["p", "q"]
        assert "missing_col" not in out["rna"].obs
        # depth has 4 unique of 2 obs per group -> continuous -> mean
        np.testing.assert_allclose(out["atac"].obs["depth"].to_numpy(), [2.0, 6.0])

    def test_missing_group_col_raises(self):
        with pytest.raises(KeyError, match="not found in any modality"):
            AnnDataUtils.merge_mdata_on_obs_annotation(
                make_mdata(), "nope", {"rna": [], "atac": []}
            )

    def test_missing_modality_in_batch_colnames_raises(self):
        with pytest.raises(KeyError):
            AnnDataUtils.merge_mdata_on_obs_annotation(
                make_mdata(), "label", {"rna": []}
            )

    def test_does_not_mutate_input(self):
        mdata = make_mdata()
        rna_X = mdata["rna"].X.copy()
        obs_cols = list(mdata["rna"].obs.columns)
        AnnDataUtils.merge_mdata_on_obs_annotation(
            mdata, "label", {"rna": ["batch"], "atac": []}
        )
        np.testing.assert_array_equal(mdata["rna"].X, rna_X)
        assert list(mdata["rna"].obs.columns) == obs_cols

    def test_group_col_in_both_modalities_uses_own_labels(self):
        mdata = make_mdata()
        mdata["atac"].obs["label"] = ["y", "y", "x", "x"]
        out = AnnDataUtils.merge_mdata_on_obs_annotation(
            mdata, "label", {"rna": [], "atac": []}
        )
        # group "x" for atac now = rows 2,3
        np.testing.assert_allclose(to_dense(out["atac"].X), [[0.0, 6.0], [2.0, 0.0]])

    def test_single_group(self):
        mdata = make_mdata()
        mdata["rna"].obs["label"] = "only"
        out = AnnDataUtils.merge_mdata_on_obs_annotation(
            mdata, "label", {"rna": [], "atac": []}
        )
        assert out["rna"].n_obs == 1
        np.testing.assert_allclose(to_dense(out["rna"].X), [[4.5, 5.5, 6.5]])

    @pytest.mark.parametrize("categorical", [False, True])
    def test_nan_group_label_skipped(self, categorical):
        # regression: a NaN label produced an all-NaN pseudo-bulk row
        mdata = make_mdata()
        labels = pd.Series(
            ["x", "x", np.nan, "y"], index=mdata["rna"].obs_names, dtype=object
        )
        mdata["rna"].obs["label"] = labels.astype("category") if categorical else labels
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no mean of empty slice
            out = AnnDataUtils.merge_mdata_on_obs_annotation(
                mdata, "label", {"rna": [], "atac": []}
            )
        for mod in ("rna", "atac"):
            assert list(out[mod].obs_names) == ["x", "y"]
            assert not np.isnan(to_dense(out[mod].X)).any()
        np.testing.assert_allclose(
            to_dense(out["rna"].X), [[1.5, 2.5, 3.5], [9.0, 10.0, 11.0]]
        )
        np.testing.assert_allclose(to_dense(out["atac"].X), [[2.0, 0.0], [0.0, 7.0]])
