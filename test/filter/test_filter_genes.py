import numpy as np
import pytest
from scipy import sparse

from cavachon.filter.filter_genes import FilterGenes
from test.filter.conftest import make_adata


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"min_cells": 3}, ["MT-CO1"]),  # boundary: 3 kept (>=)
        ({"min_cells": 2}, ["MT-CO1", "MTOR", "GAPDH", "ACTB"]),
        ({"max_cells": 2}, ["MTOR", "GAPDH", "ACTB"]),  # boundary: 2 kept
        ({"min_counts": 5}, ["GAPDH", "ACTB"]),  # boundary: 5 kept
        ({"max_counts": 4}, ["MT-CO1", "MTOR"]),
    ],
)
def test_filter_genes_keeps_expected_genes(adata, kwargs, expected):
    out = FilterGenes("fg", **kwargs)(adata)
    assert list(out.var_names) == expected
    assert out.n_obs == 4


def test_filter_genes_is_inplace(adata):
    out = FilterGenes("fg", min_cells=3)(adata)
    assert out is adata
    assert adata.n_vars == 1


def test_filter_genes_keeps_var_columns(dense_adata):
    out = FilterGenes("fg", min_counts=5)(dense_adata)
    assert list(out.var["gene"]) == ["GAPDH", "ACTB"]
    assert out.var.index.name == "RNA:gene"


def test_filter_genes_preserves_sparsity():
    out = FilterGenes("fg", min_cells=3)(make_adata(True))
    assert sparse.issparse(out.X)
    np.testing.assert_array_equal(
        out.X.toarray(), FilterGenes("fg", min_cells=3)(make_adata(False)).X
    )


def test_filter_genes_empty_result_raises(adata):
    with pytest.raises(RuntimeError, match=r"No variables left.*FilterGenes \(fg\)"):
        FilterGenes("fg", min_cells=100)(adata)


def test_filter_genes_then_cells_chaining(dense_adata):
    from cavachon.filter.filter_cells import FilterCells

    out = FilterGenes("g", min_counts=5)(dense_adata)  # keep GAPDH, ACTB
    out = FilterCells("c", min_counts=3)(out)  # c0=3, c1=0, c2=2, c3=7
    assert list(out.obs_names) == ["c0", "c3"]
    assert list(out.var_names) == ["GAPDH", "ACTB"]
