"""Pytest scaffold for HierarchicalDifferentialAnalysis.

The class does not yet exist; tests that depend on it are marked skip/xfail.

Run with:
    .pixi/envs/default/bin/python -m pytest -q test_hierarchical_differential_analysis.py
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tensorflow as tf

# ---------------------------------------------------------------------------
# Optional import — class does not exist yet
# ---------------------------------------------------------------------------

try:
    from cavachon.tools import HierarchicalDifferentialAnalysis  # type: ignore

    _HDA_AVAILABLE = True
except ImportError:
    HierarchicalDifferentialAnalysis = None  # type: ignore
    _HDA_AVAILABLE = False

skip_until_implemented = pytest.mark.skipif(
    not _HDA_AVAILABLE,
    reason="HierarchicalDifferentialAnalysis not yet implemented",
)

# ---------------------------------------------------------------------------
# Tiny synthetic fixtures (no workflow / sample.yaml needed)
# ---------------------------------------------------------------------------

import anndata as ad
import muon as mu

from cavachon.config.models.component_config import ComponentConfig
from cavachon.model.model import Model

np.random.seed(42)
tf.random.set_seed(42)

_N_CELLS = 30
_N_RNA = 8
_N_ATAC = 6
_CLUSTER_KEY = "cluster_label"


def _make_model():
    comp_a = ComponentConfig(
        name="comp_A",
        modalities=[{"name": "RNA", "distribution_names": "MultivariateNormalDiag", "n_vars": _N_RNA}],
        n_vars={"RNA": _N_RNA},
        n_vars_batch_effect={"RNA": 1},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=[],
        conditioned_on_z_hat=[],
    )
    comp_b = ComponentConfig(
        name="comp_B",
        modalities=[{"name": "ATAC", "distribution_names": "MultivariateNormalDiag", "n_vars": _N_ATAC}],
        n_vars={"ATAC": _N_ATAC},
        n_vars_batch_effect={"ATAC": 1},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=["comp_A"],
        conditioned_on_z_hat=[],
    )
    return Model.make([comp_a, comp_b])


def _make_mdata(n_cells: int = _N_CELLS) -> mu.MuData:
    rna = ad.AnnData(sp.csr_matrix(np.random.rand(n_cells, _N_RNA).astype("float32")))
    rna.obs_names = [f"cell_{i}" for i in range(n_cells)]
    rna.var_names = [f"gene_{i}" for i in range(_N_RNA)]
    # Assign two synthetic clusters
    labels = ["ClusterA" if i < n_cells // 2 else "ClusterB" for i in range(n_cells)]
    rna.obs[_CLUSTER_KEY] = labels

    atac = ad.AnnData(sp.csr_matrix(np.random.rand(n_cells, _N_ATAC).astype("float32")))
    atac.obs_names = [f"cell_{i}" for i in range(n_cells)]
    atac.var_names = [f"peak_{i}" for i in range(_N_ATAC)]
    atac.obs[_CLUSTER_KEY] = labels

    return mu.MuData({"RNA": rna, "ATAC": atac})


# Module-level cached objects
_MODEL = None
_MDATA = None


@pytest.fixture(scope="module")
def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _make_model()
    return _MODEL


@pytest.fixture(scope="module")
def mdata():
    global _MDATA
    if _MDATA is None:
        np.random.seed(42)
        tf.random.set_seed(42)
        _MDATA = _make_mdata()
    return _MDATA


@pytest.fixture(scope="module")
def hda_instance(model, mdata):
    """Instantiate HierarchicalDifferentialAnalysis (skips if unavailable)."""
    if not _HDA_AVAILABLE:
        pytest.skip("HierarchicalDifferentialAnalysis not yet implemented")
    return HierarchicalDifferentialAnalysis(
        mdata=mdata,
        model=model,
        batch_effect_colnames={},
        distribution_names={"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"},
        batch_effect_encoders={},
    )


# ---------------------------------------------------------------------------
# Test 1 — importability
# ---------------------------------------------------------------------------


def test_class_is_importable():
    """HierarchicalDifferentialAnalysis must be importable from cavachon.tools."""
    if not _HDA_AVAILABLE:
        pytest.skip("HierarchicalDifferentialAnalysis not yet implemented — expected during scaffolding")
    from cavachon.tools import HierarchicalDifferentialAnalysis as HDA  # noqa: F401

    assert HDA is not None


# ---------------------------------------------------------------------------
# Test 2 — legacy DEG column schema preserved
# ---------------------------------------------------------------------------


@skip_until_implemented
def test_preserves_legacy_deg_schema(hda_instance, mdata):
    """between_clusters output must have the 6 legacy DEG columns as first 6 columns."""
    np.random.seed(42)
    tf.random.set_seed(42)

    df = hda_instance.between_clusters(
        donor_cluster="ClusterA",
        recipient_cluster="ClusterB",
        component="comp_A",
        modality="RNA",
        n_samples=5,
        seed=42,
        batch_size=8,
    )

    expected_first_six = ["Mean(A)", "Mean(B)", "P(A>B|Z)", "P(B>A|Z)", "K(A>B|Z)", "K(B>A|Z)"]
    assert isinstance(df, pd.DataFrame), "between_clusters must return a DataFrame"
    assert list(df.columns[:6]) == expected_first_six, (
        f"First 6 columns mismatch. Got: {list(df.columns[:6])}"
    )


# ---------------------------------------------------------------------------
# Test 3 — cluster-level Z substitution changes results vs. non-substituted
# ---------------------------------------------------------------------------


@skip_until_implemented
def test_cluster_level_z_substitution_changes_results(hda_instance, mdata):
    """Substituted result should differ from a same-cluster (identity) baseline."""
    np.random.seed(42)
    tf.random.set_seed(42)

    # Same donor == recipient: identity substitution — used as "no-op" baseline
    df_identity = hda_instance.between_clusters(
        donor_cluster="ClusterA",
        recipient_cluster="ClusterA",
        component="comp_A",
        modality="RNA",
        n_samples=5,
        seed=42,
        batch_size=8,
    )

    np.random.seed(42)
    tf.random.set_seed(42)

    # Cross-cluster substitution
    df_cross = hda_instance.between_clusters(
        donor_cluster="ClusterA",
        recipient_cluster="ClusterB",
        component="comp_A",
        modality="RNA",
        n_samples=5,
        seed=42,
        batch_size=8,
    )

    assert isinstance(df_cross, pd.DataFrame)
    # At least one numeric column must differ between the two results
    numeric_cols = ["Mean(A)", "Mean(B)", "P(A>B|Z)", "P(B>A|Z)"]
    for col in numeric_cols:
        if col in df_identity.columns and col in df_cross.columns:
            if not np.allclose(df_identity[col].values, df_cross[col].values, atol=1e-6):
                return  # at least one column differs — test passes
    pytest.fail(
        "Cross-cluster substitution produced identical results to identity substitution. "
        "Substitution logic may be a no-op."
    )


# ---------------------------------------------------------------------------
# Test 4 — seeded substitution is deterministic
# ---------------------------------------------------------------------------


@skip_until_implemented
def test_seeded_substitution_is_deterministic(hda_instance, mdata):
    """Two calls with the same seed must return byte-identical DataFrames."""
    np.random.seed(42)
    tf.random.set_seed(42)
    df1 = hda_instance.between_clusters(
        donor_cluster="ClusterA",
        recipient_cluster="ClusterB",
        component="comp_A",
        modality="RNA",
        n_samples=5,
        seed=7,
        batch_size=8,
    )

    np.random.seed(42)
    tf.random.set_seed(42)
    df2 = hda_instance.between_clusters(
        donor_cluster="ClusterA",
        recipient_cluster="ClusterB",
        component="comp_A",
        modality="RNA",
        n_samples=5,
        seed=7,
        batch_size=8,
    )

    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_exact=False,
        atol=1e-5,
        obj="Seeded runs must be deterministic",
    )


# ---------------------------------------------------------------------------
# Test 5 — missing donor cluster raises ValueError
# ---------------------------------------------------------------------------


@skip_until_implemented
def test_missing_donor_cluster_raises(hda_instance, mdata):
    """Passing an invalid donor cluster name must raise ValueError."""
    with pytest.raises(ValueError, match="(?i)donor|cluster|not found|invalid"):
        hda_instance.between_clusters(
            donor_cluster="NONEXISTENT_CLUSTER_XYZ",
            recipient_cluster="ClusterB",
            component="comp_A",
            modality="RNA",
            n_samples=5,
            seed=42,
            batch_size=8,
        )


# ---------------------------------------------------------------------------
# Test 6 (optional) — empty recipient group raises ValueError
# ---------------------------------------------------------------------------


@skip_until_implemented
def test_empty_recipient_group_raises(hda_instance, mdata):
    """Passing an invalid recipient cluster name must raise ValueError."""
    with pytest.raises(ValueError, match="(?i)recipient|cluster|not found|invalid|empty"):
        hda_instance.between_clusters(
            donor_cluster="ClusterA",
            recipient_cluster="NONEXISTENT_RECIPIENT_XYZ",
            component="comp_A",
            modality="RNA",
            n_samples=5,
            seed=42,
            batch_size=8,
        )
