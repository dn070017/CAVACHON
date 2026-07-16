"""QA tests for DifferentialAnalysis.compute_x_means() and between_two_groups().

Run with:
    .pixi/envs/default/bin/python test_differential_analysis.py

All tests print PASS/FAIL and a final summary.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from cavachon.config.models.component_config import ComponentConfig
from cavachon.dataloader.dataloader import DataLoader
from cavachon.model.model import Model
from cavachon.tools.differential_analysis import DifferentialAnalysis

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    _results.append(condition)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_model():
    """Model with n_vars_batch_effect=1 to match zero-padded DataLoader output."""
    comp_A_config = ComponentConfig(
        name="comp_A",
        modalities=[{"name": "RNA", "distribution_names": "MultivariateNormalDiag", "n_vars": 8}],
        n_vars={"RNA": 8},
        n_vars_batch_effect={"RNA": 1},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=[],
        conditioned_on_z_hat=[],
    )
    comp_B_config = ComponentConfig(
        name="comp_B",
        modalities=[{"name": "ATAC", "distribution_names": "MultivariateNormalDiag", "n_vars": 6}],
        n_vars={"ATAC": 6},
        n_vars_batch_effect={"ATAC": 1},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=["comp_A"],
        conditioned_on_z_hat=[],
    )
    return Model.make([comp_A_config, comp_B_config])


def make_mdata(n_cells=20):
    """Synthetic MuData with RNA (8 vars) and ATAC (6 vars)."""
    rna = ad.AnnData(sp.csr_matrix(np.random.rand(n_cells, 8).astype("float32")))
    rna.obs_names = [f"cell_{i}" for i in range(n_cells)]
    rna.var_names = [f"gene_{i}" for i in range(8)]
    atac = ad.AnnData(sp.csr_matrix(np.random.rand(n_cells, 6).astype("float32")))
    atac.obs_names = [f"cell_{i}" for i in range(n_cells)]
    atac.var_names = [f"peak_{i}" for i in range(6)]
    return mu.MuData({"RNA": rna, "ATAC": atac})


_model = None
_mdata = None


def get_model():
    global _model
    if _model is None:
        _model = make_model()
    return _model


def get_mdata():
    global _mdata
    if _mdata is None:
        np.random.seed(0)
        _mdata = make_mdata(n_cells=20)
    return _mdata


def make_analysis():
    """DifferentialAnalysis with no explicit batch effects (DataLoader uses zero tensors)."""
    return DifferentialAnalysis(
        mdata=get_mdata(),
        model=get_model(),
        batch_effect_colnames={},
        distribution_names={"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"},
        batch_effect_encoders={},
    )


def make_batch_effect_dict(mdata, batch_size=5):
    """Build the batch_effect dict the same way between_two_groups does."""
    distribution_names = {"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"}
    dataloader = DataLoader(mdata, batch_size, {}, distribution_names, {})
    from cavachon.environment.constants import Constants
    batch_effect = dict()
    for batch in dataloader:
        for modality_name in mdata.mod.keys():
            if modality_name not in batch_effect:
                batch_effect[modality_name] = []
            key = f"{modality_name}_{Constants.TENSOR_NAME_BATCH}"
            batch_effect[modality_name].append(batch.get(key))
    for modality_name in batch_effect.keys():
        batch_effect[modality_name] = tf.concat(batch_effect[modality_name], axis=0)
    return batch_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_x_means_happy_path():
    print("\n[Test 1] compute_x_means() happy path")
    analysis = make_analysis()
    mdata = get_mdata()
    batch_effect = make_batch_effect_dict(mdata)

    distribution_names = {"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"}
    dataloader = DataLoader(mdata, 5, {}, distribution_names, {})

    np.random.seed(42)
    result = analysis.compute_x_means(
        dataset=dataloader.dataset,
        component="comp_A",
        modality="RNA",
        batch_effect=batch_effect,
        training=False,
        batch_size=5,
    )

    check("returns numpy array", isinstance(result, np.ndarray))
    check("array is 2D", result.ndim == 2, f"ndim={result.ndim}")
    check("correct number of vars (8)", result.shape[1] == 8, f"shape={result.shape}")
    check("all finite values", np.all(np.isfinite(result)), f"has_nan={np.any(np.isnan(result))}")


def test_between_two_groups_happy_path():
    print("\n[Test 2] between_two_groups() happy path")
    analysis = make_analysis()
    mdata = get_mdata()
    obs_names = list(mdata["RNA"].obs_names)
    group_a = obs_names[:10]
    group_b = obs_names[10:]

    np.random.seed(42)
    tf.random.set_seed(42)
    df = analysis.between_two_groups(
        group_a_index=group_a,
        group_b_index=group_b,
        component="comp_A",
        modality="RNA",
        z_sampling_size=2,
        x_sampling_size=5,
        batch_size=5,
    )

    expected_cols = ["Mean(A)", "Mean(B)", "P(A>B|Z)", "P(B>A|Z)", "K(A>B|Z)", "K(B>A|Z)"]
    check("returns DataFrame", isinstance(df, pd.DataFrame))
    check("correct number of rows (= n_vars = 8)", len(df) == 8, f"len={len(df)}")
    check("all expected columns present", list(df.columns) == expected_cols, f"cols={list(df.columns)}")
    check("all finite values", df.select_dtypes("number").apply(lambda c: np.all(np.isfinite(c))).all())


def test_between_two_groups_column_schema():
    print("\n[Test 3] between_two_groups() column schema")
    analysis = make_analysis()
    mdata = get_mdata()
    obs_names = list(mdata["RNA"].obs_names)
    group_a = obs_names[:10]
    group_b = obs_names[10:]

    np.random.seed(1)
    tf.random.set_seed(1)
    df = analysis.between_two_groups(
        group_a_index=group_a,
        group_b_index=group_b,
        component="comp_A",
        modality="RNA",
        z_sampling_size=2,
        x_sampling_size=5,
        batch_size=5,
    )

    expected_cols = ["Mean(A)", "Mean(B)", "P(A>B|Z)", "P(B>A|Z)", "K(A>B|Z)", "K(B>A|Z)"]
    check("exact column names match", list(df.columns) == expected_cols,
          f"got={list(df.columns)}")
    check("P(A>B|Z) in [0,1]", df["P(A>B|Z)"].between(0.0, 1.0).all())
    check("P(B>A|Z) in [0,1]", df["P(B>A|Z)"].between(0.0, 1.0).all())


def test_compute_x_means_edge_singleton_group():
    print("\n[Test 4] compute_x_means() edge case — singleton group (x_sampling_size=1)")
    analysis = make_analysis()
    mdata = get_mdata()
    batch_effect = make_batch_effect_dict(mdata)

    # Build a single-cell mdata by calling sample_mdata_x
    obs_names = list(mdata["RNA"].obs_names)
    np.random.seed(99)
    sampled_mdata = analysis.sample_mdata_x(index=obs_names[:1], x_sampling_size=1)

    distribution_names = {"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"}
    dl = DataLoader(sampled_mdata, 1, {}, distribution_names, {})

    try:
        result = analysis.compute_x_means(
            dataset=dl.dataset,
            component="comp_A",
            modality="RNA",
            batch_effect=batch_effect,
            training=False,
            batch_size=1,
        )
        check("does not crash with singleton group", True)
        check("result is 2D numpy array", isinstance(result, np.ndarray) and result.ndim == 2,
              f"shape={result.shape}")
    except Exception as e:
        check("does not crash with singleton group", False, str(e))
        check("result is 2D numpy array", False)


def test_between_two_groups_deterministic_training_false():
    print("\n[Test 5] compute_x_means() is deterministic with training=False and fixed numpy seed")
    analysis = make_analysis()
    mdata = get_mdata()
    batch_effect = make_batch_effect_dict(mdata)

    distribution_names = {"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"}
    dataloader = DataLoader(mdata, 5, {}, distribution_names, {})

    np.random.seed(77)
    result_1 = analysis.compute_x_means(
        dataset=dataloader.dataset,
        component="comp_A",
        modality="RNA",
        batch_effect=batch_effect,
        training=False,
        batch_size=5,
    )

    np.random.seed(77)
    result_2 = analysis.compute_x_means(
        dataset=dataloader.dataset,
        component="comp_A",
        modality="RNA",
        batch_effect=batch_effect,
        training=False,
        batch_size=5,
    )

    check("two runs with same numpy seed produce near-identical results (atol=1e-2)",
          np.allclose(result_1, result_2, rtol=0, atol=1e-2),
          f"max_diff={np.max(np.abs(result_1 - result_2)):.6g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DifferentialAnalysis QA tests")
    print("=" * 60)

    # --- Baseline capture ---
    print("\n[Baseline] Capturing pre-refactor baseline artifact...")
    np.random.seed(42)
    tf.random.set_seed(42)

    _baseline_mdata = make_mdata(n_cells=20)
    _baseline_model = get_model()
    _baseline_analysis = DifferentialAnalysis(
        mdata=_baseline_mdata,
        model=_baseline_model,
        batch_effect_colnames={},
        distribution_names={"RNA": "MultivariateNormalDiag", "ATAC": "MultivariateNormalDiag"},
        batch_effect_encoders={},
    )
    _obs_names = list(_baseline_mdata["RNA"].obs_names)
    _baseline_df = _baseline_analysis.between_two_groups(
        group_a_index=_obs_names[:10],
        group_b_index=_obs_names[10:],
        component="comp_A",
        modality="RNA",
        z_sampling_size=2,
        x_sampling_size=10,
        batch_size=5,
    )
    _evidence_dir = os.path.join(os.path.dirname(__file__), ".sisyphus", "evidence")
    os.makedirs(_evidence_dir, exist_ok=True)
    _baseline_path = os.path.join(_evidence_dir, "task-1-baseline.json")
    _baseline_df.to_json(_baseline_path)
    print(f"Baseline saved to {_baseline_path}")

    # --- Run tests ---
    test_compute_x_means_happy_path()
    test_between_two_groups_happy_path()
    test_between_two_groups_column_schema()
    test_compute_x_means_edge_singleton_group()
    test_between_two_groups_deterministic_training_false()

    passed = sum(_results)
    total = len(_results)
    print("\n" + "=" * 60)
    if passed == total:
        print(f"\033[92mAll {total} checks passed.\033[0m")
    else:
        print(f"\033[91m{total - passed}/{total} checks FAILED.\033[0m")
    print("=" * 60)
