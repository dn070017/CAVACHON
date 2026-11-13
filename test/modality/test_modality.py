from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tensorflow as tf

from cavachon.modality import Modality

N_OBS, N_VARS = 10, 5


@pytest.fixture
def obs():
    return pd.DataFrame(
        {"batch": ["a", "b"] * (N_OBS // 2)}, index=[f"c{i}" for i in range(N_OBS)]
    )


@pytest.fixture
def X():
    return np.arange(N_OBS * N_VARS, dtype=np.float32).reshape(N_OBS, N_VARS)


@pytest.mark.parametrize(
    "convert",
    [lambda x: x, sp.csr_matrix, lambda x: tf.constant(x), lambda x: pd.DataFrame(x)],
    ids=["ndarray", "csr", "tf.Tensor", "DataFrame"],
)
def test_construct_from_matrix_types(X, convert):
    m = Modality(convert(X), name="rna", modality_type="rna")
    assert isinstance(m, ad.AnnData)
    assert m.shape == (N_OBS, N_VARS)
    Xm = m.X.toarray() if sp.issparse(m.X) else np.asarray(m.X)
    np.testing.assert_array_equal(Xm, X)


def test_sparse_stays_sparse(X):
    m = Modality(sp.csr_matrix(X), name="rna", modality_type="rna")
    assert sp.issparse(m.X)


def test_tf_tensor_converted_to_numpy(X):
    m = Modality(tf.constant(X), name="rna", modality_type="rna")
    assert isinstance(m.X, np.ndarray)
    assert m.X.dtype == np.float32


def test_cavachon_config_in_uns(X):
    m = Modality(X, name="atac", modality_type="atac", batch_effect_colnames=["batch"])
    assert m.uns["cavachon"] == {
        "name": "atac",
        "modality_type": "atac",
        "distribution": "IndependentBernoulli",
        "batch_effect_colnames": ["batch"],
    }


@pytest.mark.parametrize(
    "modality_type,expected",
    [
        ("atac", "IndependentBernoulli"),
        ("ATAC", "IndependentBernoulli"),
        ("rna", "IndependentZeroInflatedNegativeBinomial"),
        ("Rna", "IndependentZeroInflatedNegativeBinomial"),
        ("protein", None),
    ],
)
def test_default_distribution(X, modality_type, expected):
    m = Modality(X, name="m", modality_type=modality_type)
    assert m.uns["cavachon"]["distribution"] == expected
    assert m.uns["cavachon"]["modality_type"] == modality_type


def test_explicit_distribution_overrides_default(X):
    m = Modality(X, name="m", modality_type="rna", distribution_name="MultivariateNormalDiag")
    assert m.uns["cavachon"]["distribution"] == "MultivariateNormalDiag"


def test_kwargs_forwarded_to_anndata(X, obs):
    var = pd.DataFrame(index=[f"g{j}" for j in range(N_VARS)])
    m = Modality(X, name="m", modality_type="rna", obs=obs, var=var)
    assert list(m.obs_names) == list(obs.index)
    assert list(m.var_names) == list(var.index)
    assert list(m.obs["batch"]) == list(obs["batch"])


def test_user_uns_is_merged(X):
    m = Modality(X, name="m", modality_type="rna", uns={"other": 1})
    assert m.uns["other"] == 1
    assert m.uns["cavachon"]["name"] == "m"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the caller's `uns` dict is mutated in place (cavachon config is injected "
    "into it via uns.update)",
)
def test_user_uns_dict_not_mutated(X):
    uns = {"other": 1}
    Modality(X, name="m", modality_type="rna", uns=uns)
    assert uns == {"other": 1}


def test_from_anndata_preserves_annotations(X, obs):
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs.copy())
    adata.var["gene_type"] = "x"
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["emb"] = np.zeros((N_OBS, 2))
    m = Modality(adata, name="m", modality_type="rna")
    assert isinstance(m, Modality)
    assert list(m.obs_names) == list(obs.index)
    assert list(m.obs["batch"]) == list(obs["batch"])
    assert "gene_type" in m.var
    assert "counts" in m.layers
    assert "emb" in m.obsm
    assert m.uns["cavachon"]["name"] == "m"
    np.testing.assert_array_equal(m.X.toarray(), X)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: wrapping an AnnData mutates the input's .uns (X.uns.update(uns))",
)
def test_from_anndata_does_not_mutate_input(X, obs):
    adata = ad.AnnData(X=X.copy(), obs=obs.copy())
    Modality(adata, name="m", modality_type="rna")
    assert "cavachon" not in adata.uns


def test_write_h5ad_delegates_with_cavachon_config(monkeypatch, X, obs):
    """No real file: the anndata writer is mocked where AnnData.write_h5ad looks it up."""
    from pathlib import Path

    writer = mock.MagicMock()
    monkeypatch.setattr("anndata.io.write_h5ad", writer)
    m = Modality(sp.csr_matrix(X), name="m", modality_type="rna", obs=obs)
    m.write_h5ad("m.h5ad")
    writer.assert_called_once()
    path, written = writer.call_args.args
    assert path == Path("m.h5ad")
    assert written is m
    assert written.uns["cavachon"]["distribution"] == "IndependentZeroInflatedNegativeBinomial"


def test_wraps_anndata_returned_by_reader(monkeypatch, X, obs):
    """Typical usage Modality(anndata.read_h5ad(path), ...) with the reader mocked."""
    reader = mock.MagicMock(return_value=ad.AnnData(X=sp.csr_matrix(X), obs=obs.copy()))
    monkeypatch.setattr(ad, "read_h5ad", reader)
    m = Modality(ad.read_h5ad("in.h5ad"), name="rna", modality_type="rna")
    reader.assert_called_once_with("in.h5ad")
    assert list(m.obs_names) == list(obs.index)
    assert m.uns["cavachon"]["name"] == "rna"
