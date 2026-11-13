import warnings

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cavachon.modality import Modality, MultiModality


def make(names, n_vars=3, value_offset=0.0):
    """AnnData whose row values encode the cell id (c<k> -> k + offset)."""
    ids = np.array([int(n[1:]) for n in names], dtype=np.float32)
    X = np.repeat(ids[:, None], n_vars, axis=1) + value_offset
    obs = pd.DataFrame({"batch": ["a", "b"] * (len(names) // 2) + ["a"] * (len(names) % 2)},
                       index=list(names))
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    return adata


def cell_ids(adata):
    return adata.X.toarray()[:, 0].astype(int).tolist()


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


NAMES = [f"c{i}" for i in range(10)]


def test_from_mapping_same_cells():
    mm = MultiModality({"rna": make(NAMES), "atac": make(NAMES, n_vars=4)})
    assert isinstance(mm, mu.MuData)
    assert mm.n_obs == 10
    assert set(mm.mod) == {"rna", "atac"}
    assert list(mm["rna"].obs_names) == NAMES
    assert list(mm["atac"].obs_names) == NAMES


def test_intersection_drops_cells_not_shared():
    mm = MultiModality({"rna": make(NAMES), "atac": make(NAMES[3:])})
    assert mm.n_obs == 7
    assert list(mm["rna"].obs_names) == NAMES[3:]
    assert list(mm["atac"].obs_names) == NAMES[3:]


def test_order_follows_first_modality_and_rows_follow_names():
    rev = NAMES[::-1]
    mm = MultiModality({"rna": make(NAMES), "atac": make(rev[:7])})
    expected = [n for n in NAMES if n in rev[:7]]
    assert list(mm["rna"].obs_names) == expected
    assert list(mm["atac"].obs_names) == expected
    # X rows were permuted together with obs names
    assert cell_ids(mm["rna"]) == [int(n[1:]) for n in expected]
    assert cell_ids(mm["atac"]) == [int(n[1:]) for n in expected]
    assert list(mm.obs_names) == expected


def test_obs_columns_follow_reordering():
    rna = make(NAMES)
    atac = make(NAMES[::-1])
    atac.obs["tag"] = atac.obs_names
    mm = MultiModality({"rna": rna, "atac": atac})
    assert list(mm["atac"].obs["tag"]) == list(mm["atac"].obs_names)


def test_from_single_modality_uses_its_name():
    m = Modality(make(NAMES), name="protein", modality_type="protein")
    mm = MultiModality(m)
    assert list(mm.mod) == ["protein"]
    assert mm["protein"].uns["cavachon"]["name"] == "protein"


def test_from_plain_anndata():
    mm = MultiModality(make(NAMES))
    assert mm.n_obs == 10
    assert len(mm.mod) == 1


def test_from_mudata_intersects():
    md = mu.MuData({"rna": make(NAMES), "atac": make(NAMES[2:])})
    mm = MultiModality(md)
    assert mm.n_obs == 8
    assert list(mm["rna"].obs_names) == NAMES[2:]


def test_uns_config_preserved():
    a = Modality(make(NAMES), name="atac", modality_type="atac")
    r = Modality(make(NAMES), name="rna", modality_type="rna")
    mm = MultiModality({"atac": a, "rna": r})
    assert mm["atac"].uns["cavachon"]["distribution"] == "IndependentBernoulli"
    assert mm["rna"].uns["cavachon"]["distribution"] == "IndependentZeroInflatedNegativeBinomial"


def test_shared_var_names_across_modalities_kept_per_modality():
    mm = MultiModality({"rna": make(NAMES, n_vars=3), "atac": make(NAMES, n_vars=3)})
    assert mm.n_vars == 6
    assert list(mm["rna"].var_names) == ["g0", "g1", "g2"]
    assert list(mm["atac"].var_names) == ["g0", "g1", "g2"]


def test_empty_intersection_yields_zero_cells():
    """Documents current behaviour: no error, silently empty."""
    mm = MultiModality({"rna": make(NAMES), "atac": make(["c100", "c101"])})
    assert mm.n_obs == 0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the caller's mapping is mutated in place (entries replaced by subset copies)",
)
def test_input_mapping_not_mutated():
    rna = make(NAMES)
    data = {"rna": rna, "atac": make(NAMES[3:])}
    MultiModality(data)
    assert data["rna"] is rna
    assert data["rna"].n_obs == 10


@pytest.mark.xfail(
    strict=True,
    reason="BUG: passing a MuData drops non-shared cells from the caller's MuData in place "
    "(muon.pp.intersect_obs + writes into data.mod)",
)
def test_input_mudata_not_mutated():
    md = mu.MuData({"rna": make(NAMES), "atac": make(NAMES[3:])})
    MultiModality(md)
    assert md["rna"].n_obs == 10


@pytest.mark.xfail(
    strict=True,
    reason="BUG: duplicated obs_names in one modality crash with an opaque pandas "
    "InvalidIndexError from the intersection/reindex step instead of being handled or "
    "rejected with a clear message",
)
def test_duplicate_obs_names_keep_modalities_same_length():
    names = ["c0", "c1", "c1", "c2"]
    rna = make(names)
    atac = make(["c0", "c1", "c2"])
    mm = MultiModality({"rna": rna, "atac": atac})
    assert mm["rna"].n_obs == mm["atac"].n_obs
