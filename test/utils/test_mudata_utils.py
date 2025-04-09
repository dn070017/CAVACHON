import numpy as np
import pandas as pd
from anndata import AnnData
from muon import MuData

from cavachon.environment.constants import Constants
from cavachon.utils.mudata_utils import MuDataUtils


def test_check_if_cavachon_config_exists():
    adata = AnnData(np.random.rand(10, 5))
    mdata = MuData({"A": adata})
    assert not MuDataUtils.check_if_cavachon_config_exists(mdata, "test_field")

    mdata.uns[Constants.MDATA_UNS_FIELD_CAVACHON] = {"test_field": "test_value"}
    assert MuDataUtils.check_if_cavachon_config_exists(mdata, "test_field")


def test_merge_mdata_on_obs_annotation():
    adata_a = AnnData(np.random.rand(10, 5))
    adata_a.obs["group"] = ["G1"] * 5 + ["G2"] * 5
    adata_a.obs["batch"] = ["B1"] * 3 + ["B2"] * 7
    adata_b = AnnData(np.random.rand(10, 3))
    adata_b.obs["group"] = ["G1"] * 5 + ["G2"] * 5
    adata_b.obs["batch"] = ["B1"] * 4 + ["B2"] * 6
    mdata = MuData({"A": adata_a, "B": adata_b})

    batch_effect_colnames = {"A": ["batch"], "B": ["batch"]}
    mdata = MuDataUtils.merge_mdata_on_obs_annotation(
        mdata, "group", batch_effect_colnames
    )
    assert len(mdata.obs) == 2
    assert "A" in mdata.mod.keys()
    assert "B" in mdata.mod.keys()
    assert mdata["A"].obs.index.tolist() == ["G1", "G2"]
    assert mdata["B"].obs.index.tolist() == ["G1", "G2"]


def test_reorder_or_filter_adata_obs():
    adata = AnnData(np.random.rand(10, 5))
    adata.obs.index = pd.Index(
        ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    )

    obs_index = pd.Index(["C2", "C1", "C3"])
    adata_reordered = MuDataUtils.reorder_or_filter_adata_obs(adata, obs_index)
    assert adata_reordered.obs.index.tolist() == ["C2", "C1", "C3"]
