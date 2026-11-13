import numpy as np
import pytest
from scipy import sparse

from cavachon.filter.filter_cells import FilterCells
from test.filter.conftest import make_adata


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"min_counts": 4}, ["c0", "c2", "c3"]),  # boundary: 4 kept (>=)
        ({"min_counts": 5}, ["c2", "c3"]),
        ({"max_counts": 6}, ["c0", "c1", "c2"]),  # boundary: 6 kept (<=)
        ({"min_genes": 3}, ["c2", "c3"]),
        ({"max_genes": 2}, ["c0", "c1"]),
        ({"min_genes": 0}, ["c0", "c1", "c2", "c3"]),
    ],
)
def test_filter_cells_keeps_expected_cells(adata, kwargs, expected):
    out = FilterCells("fc", **kwargs)(adata)
    assert list(out.obs_names) == expected
    assert out.n_vars == 4


def test_filter_cells_is_inplace_and_returns_same_object(adata):
    out = FilterCells("fc", min_counts=5)(adata)
    assert out is adata
    assert adata.n_obs == 2


def test_filter_cells_adds_qc_column(dense_adata):
    out = FilterCells("fc", min_counts=1)(dense_adata)
    assert "n_counts" in out.obs
    np.testing.assert_array_equal(out.obs["n_counts"].values, [4, 6, 10])


def test_filter_cells_preserves_sparsity():
    adata = make_adata(is_sparse=True)
    out = FilterCells("fc", min_counts=1)(adata)
    assert sparse.issparse(out.X)


def test_filter_cells_dense_and_sparse_agree():
    d = FilterCells("fc", min_genes=2)(make_adata(False))
    s = FilterCells("fc", min_genes=2)(make_adata(True))
    assert list(d.obs_names) == list(s.obs_names)
    np.testing.assert_array_equal(d.X, s.X.toarray())


def test_filter_cells_empty_result_raises(adata):
    with pytest.raises(RuntimeError, match=r"FilterCells \(fc\)"):
        FilterCells("fc", min_counts=1000)(adata)


def test_filter_cells_rejects_multiple_criteria(dense_adata):
    # scanpy only accepts one criterion per call; chain steps instead.
    with pytest.raises(ValueError):
        FilterCells("fc", min_counts=1, min_genes=1)(dense_adata)


def test_filter_cells_requires_a_criterion(dense_adata):
    with pytest.raises(ValueError):
        FilterCells("fc")(dense_adata)


def test_filter_cells_chaining(dense_adata):
    out = FilterCells("a", min_counts=1)(dense_adata)
    out = FilterCells("b", max_genes=3)(out)
    assert list(out.obs_names) == ["c0", "c2"]


def test_filter_cells_on_view(dense_adata):
    view = dense_adata[["c0", "c1", "c2"]]
    assert view.is_view
    out = FilterCells("fc", min_counts=1)(view)
    assert list(out.obs_names) == ["c0", "c2"]
    # parent untouched
    assert dense_adata.n_obs == 4


def test_filter_cells_is_reusable(dense_adata):
    f = FilterCells("fc", min_counts=5)
    f(dense_adata)
    other = make_adata()
    f(other)
    assert other.n_obs == 2
    assert f.kwargs == {"min_counts": 5, "inplace": True}
