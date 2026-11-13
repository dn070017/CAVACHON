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


# --------------------------------------------------------------------- bugs


# regression: the default 'MT' qc_var used to be the unanchored,
# case-sensitive prefix "MT", which also flagged nuclear genes like MTOR.
def test_default_mt_qc_var_matches_only_mitochondrial_genes(dense_adata):
    qc([])(dense_adata)
    # only MT-CO1 is mitochondrial
    assert list(dense_adata.var["MT"]) == [True, False, False, False]
    # c3: MT-CO1=1 of 10 counts; MTOR counts must not count as mitochondrial
    assert dense_adata.obs.loc["c3", "pct_counts_MT"] == pytest.approx(10.0)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: FilterQC crashes (AttributeError) when var index has no name",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG: filter_threshold documented as optional but omitting it "
    "raises KeyError",
)
def test_filter_threshold_optional(dense_adata):
    out = FilterQC("qc", percent_top=None)(dense_adata)
    assert out.n_obs == 4


@pytest.mark.xfail(
    strict=True,
    reason="BUG: inherited scanpy default percent_top=(50,100,200,500) makes "
    "FilterQC crash (IndexError) on modalities with < 500 features",
)
def test_default_percent_top_on_small_feature_set(dense_adata):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = FilterQC(
            "qc",
            filter_threshold=[{"field": "total_counts", "threshold": 1}],
        )(dense_adata)
    assert out.n_obs == 3


def test_selection_mask_built_via_general_utils(dense_adata, mock_general_utils):
    qc([{"field": "total_counts", "operator": "ge", "threshold": 6}])(dense_adata)
    mock_general_utils.duplicate_obj_to_list.assert_called_once_with(True, 4)
