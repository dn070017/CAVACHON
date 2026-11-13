import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cavachon.filter.filter_cells import FilterCells
from cavachon.filter.filter_qc import FilterQC
from test.filter.conftest import COUNTS, make_adata

# cell totals: c0=4, c1=0, c2=6, c3=10


def qc(criteria, **kwargs):
    kwargs.setdefault("percent_top", None)
    return FilterQC("qc", filter_threshold=criteria, **kwargs)


@pytest.mark.parametrize(
    "op, threshold, expected",
    [
        ("ge", 6, ["c2", "c3"]),
        ("gt", 6, ["c3"]),
        ("le", 4, ["c0", "c1"]),
        ("lt", 4, ["c1"]),
        ("eq", 6, ["c2"]),
        ("ne", 6, ["c0", "c1", "c3"]),
    ],
)
def test_operators_and_boundaries(adata, op, threshold, expected):
    out = qc([{"field": "total_counts", "operator": op, "threshold": threshold}])(
        adata
    )
    assert list(out.obs_names) == expected


def test_default_operator_is_ge_and_default_threshold_is_zero(adata):
    out = qc([{"field": "total_counts", "threshold": 6}])(adata)
    assert list(out.obs_names) == ["c2", "c3"]
    out = qc([{"field": "total_counts"}])(make_adata())
    assert out.n_obs == 4


def test_multiple_criteria_are_anded(adata):
    out = qc(
        [
            {"field": "total_counts", "operator": "ge", "threshold": 4},
            {"field": "n_genes_by_counts", "operator": "lt", "threshold": 4},
        ]
    )(adata)
    assert list(out.obs_names) == ["c0", "c2"]


def test_empty_criteria_keeps_all_cells(adata):
    out = qc([])(adata)
    assert out.n_obs == 4


def test_pct_counts_with_explicit_qc_var(adata):
    out = qc(
        [{"field": "pct_counts_MT-", "operator": "le", "threshold": 30}],
        qc_vars=["MT-"],
    )(adata)
    # pct MT-: c0=25, c1=NaN, c2=33.3, c3=10
    assert list(out.obs_names) == ["c0", "c3"]
    np.testing.assert_allclose(out.obs["pct_counts_MT-"].values, [25.0, 10.0])


def test_nan_metric_cells_are_dropped(adata):
    # an all-zero cell has NaN percentage; NaN compares False for every op
    out = qc(
        [{"field": "pct_counts_MT-", "operator": "le", "threshold": 100}],
        qc_vars=["MT-"],
    )(adata)
    assert "c1" not in out.obs_names


def test_qc_var_flags_written_to_var(dense_adata):
    qc([], qc_vars=["MT-", "GAP"])(dense_adata)
    assert list(dense_adata.var["MT-"]) == [True, False, False, False]
    assert list(dense_adata.var["GAP"]) == [False, False, True, False]


def test_missing_field_warns_and_is_ignored(adata):
    with pytest.warns(RuntimeWarning, match="not_a_column is not in the obs_df"):
        out = qc(
            [
                {"field": "not_a_column", "operator": "ge", "threshold": 1e9},
                {"field": "total_counts", "operator": "gt", "threshold": 6},
            ]
        )(adata)
    assert list(out.obs_names) == ["c3"]


def test_misspelled_field_key_warns(dense_adata):
    with pytest.warns(RuntimeWarning, match="None is not in the obs_df"):
        out = qc([{"fields": "total_counts", "threshold": 1e9}])(dense_adata)
    assert out.n_obs == 4


def test_unknown_operator_raises(dense_adata):
    with pytest.raises(AttributeError):
        qc([{"field": "total_counts", "operator": "bogus", "threshold": 1}])(
            dense_adata
        )


def test_empty_result_raises(adata):
    with pytest.raises(RuntimeError, match=r"No objects left.*FilterQC \(qc\)"):
        qc([{"field": "total_counts", "operator": "gt", "threshold": 1e6}])(adata)


def test_returns_new_actual_object_not_view(dense_adata):
    out = qc([{"field": "total_counts", "operator": "gt", "threshold": 6}])(
        dense_adata
    )
    assert out is not dense_adata
    assert not out.is_view
    # input is NOT subset (only annotated with QC columns)
    assert dense_adata.n_obs == 4
    assert "total_counts" in dense_adata.obs
    out.X[0, 0] = 999
    assert dense_adata.X[3, 0] == COUNTS[3, 0]


def test_sparse_result_stays_sparse():
    out = qc([{"field": "total_counts", "operator": "ge", "threshold": 4}])(
        make_adata(True)
    )
    assert sparse.issparse(out.X)


def test_does_not_mutate_own_kwargs_and_is_reusable(dense_adata):
    criteria = [{"field": "total_counts", "operator": "ge", "threshold": 6}]
    f = qc(criteria)
    before = {k: v for k, v in f.kwargs.items()}
    f(dense_adata)
    out = f(make_adata())
    assert f.kwargs == before
    assert "qc_vars" not in f.kwargs
    assert list(out.obs_names) == ["c2", "c3"]


def test_chained_after_filter_cells(dense_adata):
    out = FilterCells("fc", min_counts=1)(dense_adata)
    out = qc([{"field": "n_genes_by_counts", "operator": "ge", "threshold": 3}])(
        out
    )
    assert list(out.obs_names) == ["c2", "c3"]


def test_running_twice_recomputes_on_filtered_data(dense_adata):
    out = qc([{"field": "total_counts", "operator": "ge", "threshold": 4}])(
        dense_adata
    )
    out = qc([{"field": "total_counts", "operator": "le", "threshold": 6}])(out)
    assert list(out.obs_names) == ["c0", "c2"]
    np.testing.assert_array_equal(out.obs["total_counts"].values, [4, 6])


# --------------------------------------------------------- default qc_vars


def _adata_with_genes(genes, counts):
    var = pd.DataFrame({"gene": genes}, index=pd.Index(genes, name="RNA:gene"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(len(counts))])
    return anndata.AnnData(X=np.asarray(counts, dtype=np.float32), obs=obs, var=var)


# regression: default qc_vars must flag only the intended features.
def test_default_qc_vars_flags_and_column_names():
    genes = ["MT-CO1", "mt-Nd1", "Mt-Atp6", "MTOR", "MTHFR", "ERCC-00002",
             "ERCC1", "GAPDH"]
    adata = _adata_with_genes(genes, [[1, 1, 1, 1, 1, 1, 1, 3]])
    FilterQC("qc", percent_top=None, filter_threshold=[])(adata)
    assert list(adata.var["MT"]) == [True, True, True, False, False, False,
                                     False, False]
    # ERCC1 is a nuclear DNA-repair gene, not a spike-in
    assert list(adata.var["ERCC"]) == [False, False, False, False, False, True,
                                       False, False]
    # QC metric names are unchanged
    assert adata.obs.loc["c0", "pct_counts_MT"] == pytest.approx(30.0)
    assert adata.obs.loc["c0", "pct_counts_ERCC"] == pytest.approx(10.0)


def test_default_qc_vars_mouse_mt_filtering():
    adata = _adata_with_genes(["mt-Co1", "Actb"], [[9, 1], [1, 9]])
    out = FilterQC(
        "qc",
        percent_top=None,
        filter_threshold=[{"field": "pct_counts_MT", "operator": "le",
                           "threshold": 50}],
    )(adata)
    assert list(out.obs_names) == ["c1"]


def test_user_qc_vars_still_unanchored_case_sensitive_prefix(dense_adata):
    # user-provided qc_vars keep the previous semantics: name == regex prefix
    qc([], qc_vars=["MT", "mt"])(dense_adata)
    assert list(dense_adata.var["MT"]) == [True, True, False, False]
    assert list(dense_adata.var["mt"]) == [False, False, False, False]


def test_default_qc_vars_not_stored_on_filter(dense_adata):
    f = qc([])
    f(dense_adata)
    assert "qc_vars" not in f.kwargs
    assert FilterQC.default_qc_vars == {"ERCC": r"^ERCC-", "MT": r"(?i)^MT-"}


# --------------------------------------------------- feature-name lookup


# regression: an unnamed var index used to raise AttributeError.
def test_works_when_var_index_unnamed():
    adata = anndata.AnnData(
        X=COUNTS.copy(),
        var=pd.DataFrame(index=["MT-CO1", "MTOR", "GAPDH", "ACTB"]),
    )
    out = qc(
        [{"field": "total_counts", "operator": "ge", "threshold": 4}],
        qc_vars=["MT-"],
    )(adata)
    assert out.n_obs == 3
    assert list(adata.var["MT-"]) == [True, False, False, False]


@pytest.mark.parametrize(
    "index_name, columns",
    [
        ("gene", {"gene": ["X", "X", "X", "X"]}),  # no ':' -> var_names
        ("RNA:symbol", {}),  # referenced column missing -> var_names
        ("RNA:symbol", {"other": ["X", "X", "X", "X"]}),
    ],
)
def test_falls_back_to_var_names(index_name, columns):
    var = pd.DataFrame(
        columns, index=pd.Index(["MT-CO1", "MTOR", "GAPDH", "ACTB"], name=index_name)
    )
    adata = anndata.AnnData(X=COUNTS.copy(), var=var)
    qc([])(adata)
    assert list(adata.var["MT"]) == [True, False, False, False]
    assert adata.obs["pct_counts_MT"].iloc[3] == pytest.approx(10.0)


def test_filereader_style_uses_referenced_column(dense_adata):
    # var index 'RNA:gene' -> patterns matched against var['gene'], not the
    # index values: make them differ to prove which one is used.
    dense_adata.var["gene"] = ["GAPDH", "ACTB", "MT-CO1", "MTOR"]
    qc([])(dense_adata)
    assert list(dense_adata.var["MT"]) == [False, False, True, False]
    assert dense_adata.var.index.name == "RNA:gene"
    # c0 counts: GAPDH(col 2)=3 of 4
    assert dense_adata.obs.loc["c0", "pct_counts_MT"] == pytest.approx(75.0)


# --------------------------------------------------------- filter_threshold


# regression: omitting filter_threshold used to raise KeyError.
def test_filter_threshold_optional(dense_adata):
    out = FilterQC("qc", percent_top=None)(dense_adata)
    assert out.n_obs == 4
    assert "pct_counts_MT" in out.obs
    assert "total_counts" in out.obs


# --------------------------------------------------------------- percent_top


# regression: scanpy's default percent_top used to raise IndexError on
# modalities with < 500 features.
def test_default_percent_top_on_small_feature_set(dense_adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = FilterQC(
            "qc",
            filter_threshold=[{"field": "total_counts", "threshold": 1}],
        )(dense_adata)
    assert out.n_obs == 3
    assert not any("pct_counts_in_top" in c for c in out.obs)


def test_default_percent_top_keeps_possible_values():
    n_vars = 120
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(n_vars)]
    var = pd.DataFrame({"gene": genes}, index=pd.Index(genes, name="RNA:gene"))
    adata = anndata.AnnData(
        X=rng.integers(1, 5, size=(3, n_vars)).astype(np.float32), var=var
    )
    f = FilterQC("qc")
    f(adata)
    top = sorted(c for c in adata.obs if c.startswith("pct_counts_in_top"))
    assert top == ["pct_counts_in_top_100_genes", "pct_counts_in_top_50_genes"]
    assert "percent_top" not in f.kwargs


def test_default_percent_top_boundary_equal_to_n_vars(monkeypatch):
    import cavachon.filter.filter_qc as mod
    from unittest.mock import MagicMock

    spy = MagicMock()
    monkeypatch.setattr(mod.scanpy.pp, "calculate_qc_metrics", spy)
    genes = [f"G{i}" for i in range(50)]
    var = pd.DataFrame({"gene": genes}, index=pd.Index(genes, name="RNA:gene"))
    FilterQC("qc", filter_threshold=[])(
        anndata.AnnData(X=np.ones((2, 50), dtype=np.float32), var=var)
    )
    assert spy.call_args.kwargs["percent_top"] == (50,)


def test_explicit_percent_top_passed_through_unchanged(dense_adata):
    # explicit bad values still surface scanpy's error
    with pytest.raises(IndexError):
        FilterQC("qc", percent_top=[50], filter_threshold=[])(dense_adata)


def test_explicit_percent_top_used(dense_adata):
    qc([], percent_top=[2])(dense_adata)
    assert "pct_counts_in_top_2_genes" in dense_adata.obs
