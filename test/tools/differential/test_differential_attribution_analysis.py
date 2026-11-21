"""Unit tests for cavachon.tools.differential_attribution_analysis.

DataLoader / ReflectionHandler are fakes (see conftest).
pd.read_csv is mocked. The FakeModel has two components:

    z_a = X @ E_a   (reads g0, g1)       z_b = X @ E_b   (reads g2)
    x   = link(z_a @ W_a + z_b @ W_b) + batch_effect

For the identity link Integrated Gradients is exact:
``attr_c[j] = (delta_z_c @ W_c)[j]``. For ``link(t) = t**2`` the left
Riemann sum used by the code has the closed form
``attr_c[j] = 2 * d_c[j] * (s0[j] + ds[j] * (n - 1) / (2 n))`` with
``d_c = delta_z_c @ W_c``, ``ds = sum_c d_c`` and ``s0`` the baseline
pre-activation.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from cavachon.tools.differential_attribution_analysis import (
    DifferentialAttributionAnalysis,
)
from test.tools.differential.conftest import (
    DAA_MODULE,
    FakeDataLoader,
    FakeModel,
    make_mdata,
)

E = {
    "a": np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32),
    "b": np.array([[0, 0], [0, 0], [1, 1]], dtype=np.float32),
}
W = {
    "a": np.array([[1, 2, 0], [0, -1, 3]], dtype=np.float32),
    "b": np.array([[2, 0, 1], [0, 1, -1]], dtype=np.float32),
}
# group A = cells 0,1 ; group B = cells 2,3,4
X = np.array([[2, 1, 3], [4, 3, 1], [1, 1, 1], [0, 2, 0], [2, 0, 2]], dtype=np.float32)
A_IDX = ["cell0", "cell1"]
B_IDX = ["cell2", "cell3", "cell4"]
FEATURES = ["g0", "g1", "g2"]


def make_daa(link=None, mdata=None, **kwargs):
    model = FakeModel(E, W, link=link)
    mdata = mdata if mdata is not None else make_mdata(X)
    return DifferentialAttributionAnalysis(mdata, model, **kwargs), model


def expected_terms():
    z = {c: X @ E[c] for c in E}
    zA = {c: z[c][:2].mean(axis=0) for c in E}
    zB = {c: z[c][2:].mean(axis=0) for c in E}
    d = {c: (zA[c] - zB[c]) @ W[c] for c in E}
    s0 = sum(zB[c] @ W[c] for c in E)
    return zA, zB, d, s0


def deg(features=FEATURES, **cols):
    return pd.DataFrame(cols or {"K(A>B|Z)": np.zeros(len(features))}, index=features)


def test_is_a_differential_analysis():
    assert issubclass(DifferentialAttributionAnalysis, DifferentialAnalysis)


# --------------------------------------------------------------------------
# compute_group_latent_mean
# --------------------------------------------------------------------------


def test_compute_group_latent_mean_across_batches():
    daa, model = make_daa()
    means = daa.compute_group_latent_mean(pd.Index(B_IDX), batch_size=2)
    _, zB, _, _ = expected_terms()
    assert set(means) == {"a", "b"}
    for c in E:
        assert tuple(means[c].shape) == (1, 2)
        np.testing.assert_allclose(means[c].numpy()[0], zB[c], rtol=1e-6)
    assert len(model.encode_calls) == 2 and not any(model.encode_calls)


def test_compute_group_latent_mean_dataloader_arguments():
    enc = {"RNA": {}}
    daa, _ = make_daa(batch_effect_encoders=enc)
    daa.compute_group_latent_mean(["cell0"], batch_size=3)
    loader = FakeDataLoader.instances[-1]
    assert list(loader.mdata["RNA"].obs_names) == ["cell0"]
    assert loader.batch_size == 3
    assert loader.batch_effect_colnames == {}
    assert loader.distribution_names == {}
    assert loader.batch_effect_encoders is enc


def test_compute_group_latent_mean_empty_index_raises():
    daa, _ = make_daa()
    with pytest.raises(ValueError, match="empty index"):
        daa.compute_group_latent_mean([])


def test_compute_group_latent_mean_unknown_cell_raises():
    daa, _ = make_daa()
    with pytest.raises(KeyError):
        daa.compute_group_latent_mean(["nope"])


# --------------------------------------------------------------------------
# compute_differential_attribution_target_batch
# --------------------------------------------------------------------------


def test_target_batch_decodes_mean(reflection):
    daa, model = make_daa(link=lambda t: t * t)
    batch = {"RNA_matrix": tf.zeros((1, 3)), "RNA_batch_effect": tf.fill((1, 1), 0.5)}
    z = {"a": tf.constant([[1.0, 2.0]]), "b": tf.constant([[0.0, 1.0]])}
    out = daa.compute_differential_attribution_target_batch(batch, "a", "RNA", z)
    pre = np.array([[1, 2]]) @ W["a"] + np.array([[0, 1]]) @ W["b"]
    np.testing.assert_allclose(out.numpy(), pre**2 + 0.5)
    assert model.decode_calls == [(["a"], False)]
    reflection.get_class_by_name.assert_called_with(
        "FakeDist", "distributions", "Distribution"
    )


# --------------------------------------------------------------------------
# compute_differential_attribution: validation
# --------------------------------------------------------------------------


def test_empty_with_respect_to_raises():
    daa, _ = make_daa()
    with pytest.raises(ValueError, match="at least one component"):
        daa.compute_differential_attribution(deg(), "a", "RNA", A_IDX, B_IDX, [])


def test_unknown_with_respect_to_raises():
    daa, _ = make_daa()
    with pytest.raises(
        ValueError, match=r"Unknown components in with_respect_to: \['x'\]"
    ):
        daa.compute_differential_attribution(
            deg(), "a", "RNA", A_IDX, B_IDX, ["a", "x"]
        )


def test_unknown_modality_raises():
    daa, _ = make_daa()
    with pytest.raises(ValueError, match="Modality 'ATAC' not found"):
        daa.compute_differential_attribution(deg(), "a", "ATAC", A_IDX, B_IDX, ["a"])


def test_no_overlapping_features_warns_and_returns_none():
    daa, model = make_daa()
    with pytest.warns(UserWarning, match="No DEG features overlap"):
        out = daa.compute_differential_attribution(
            deg(["x", "y"]), "a", "RNA", A_IDX, B_IDX, ["a"]
        )
    assert out is None
    assert model.encode_calls == []


# --------------------------------------------------------------------------
# compute_differential_attribution: numerics
# --------------------------------------------------------------------------


def test_linear_model_attributions_are_exact_and_complete():
    daa, _ = make_daa()
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], batch_size=2, n_steps=3
    )
    zA, zB, d, s0 = expected_terms()
    np.testing.assert_allclose(result["attr_a"], d["a"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result["attr_b"], d["b"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result["abs_attr_a"], np.abs(d["a"]), atol=1e-5)
    np.testing.assert_allclose(result["Delta"], d["a"] + d["b"], atol=1e-5)
    np.testing.assert_allclose(result["Expected(B)"], s0, atol=1e-5)
    np.testing.assert_allclose(result["Expected(A)"], s0 + d["a"] + d["b"], atol=1e-5)
    # completeness
    np.testing.assert_allclose(
        result["attr_a"] + result["attr_b"], result["Delta"], atol=1e-5
    )


def test_contributions_are_normalized():
    daa, _ = make_daa()
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=2
    )
    _, _, d, _ = expected_terms()
    total = d["a"] + d["b"]
    abs_total = np.abs(d["a"]) + np.abs(d["b"])
    np.testing.assert_allclose(result["contrib_a"], d["a"] / total, rtol=1e-4)
    np.testing.assert_allclose(result["contrib_b"], d["b"] / total, rtol=1e-4)
    np.testing.assert_allclose(
        result["abs_contrib_a"], np.abs(d["a"]) / abs_total, rtol=1e-4
    )
    np.testing.assert_allclose(
        result["contrib_a"] + result["contrib_b"], 1.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        result["abs_contrib_a"] + result["abs_contrib_b"], 1.0, rtol=1e-5
    )


def test_zero_change_gives_zero_contributions_not_nan():
    daa, _ = make_daa()
    # identical groups -> delta_z == 0 everywhere
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, A_IDX, ["a", "b"], n_steps=2
    )
    for col in ("attr_a", "attr_b", "Delta", "contrib_a", "abs_contrib_b"):
        np.testing.assert_allclose(result[col], 0.0)
    assert not result.isna().any().any()


def test_quadratic_link_left_riemann_closed_form():
    n = 4
    daa, _ = make_daa(link=lambda t: t * t)
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=n
    )
    _, _, d, s0 = expected_terms()
    ds = d["a"] + d["b"]
    for c in ("a", "b"):
        expected = 2 * d[c] * (s0 + ds * (n - 1) / (2 * n))
        np.testing.assert_allclose(result[f"attr_{c}"], expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result["Delta"], (s0 + ds) ** 2 - s0**2, rtol=1e-5)


def test_single_component_reports_raw_attributions_only():
    daa, _ = make_daa()
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["b"], n_steps=2
    )
    _, _, d, _ = expected_terms()
    assert "attr_a" not in result.columns
    assert not any(
        col.startswith(("contrib_", "abs_contrib_")) for col in result.columns
    )
    np.testing.assert_allclose(result["attr_b"], d["b"], atol=1e-5)
    # only component b moves to the target group
    np.testing.assert_allclose(result["Delta"], d["b"], atol=1e-5)


def test_deg_subset_order_and_extra_columns():
    daa, _ = make_daa()
    table = pd.DataFrame(
        {"Mean(A)": [3.0, 1.0, 0.0], "Mean(B)": [1.0, 1.0, 0.0], "keep": [1, 2, 3]},
        index=["g2", "unknown", "g0"],
    )
    result = daa.compute_differential_attribution(
        table, "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=2
    )
    _, _, d, _ = expected_terms()
    assert sorted(result.index) == ["g0", "g2"]
    for gene, j in (("g0", 0), ("g2", 2)):
        assert result.loc[gene, "attr_a"] == pytest.approx(d["a"][j], abs=1e-5)
        assert result.loc[gene, "attr_b"] == pytest.approx(d["b"][j], abs=1e-5)
    assert result.loc["g2", "LogFC"] == pytest.approx(np.log1p(3.0) - np.log1p(1.0))
    assert result.loc["g0", "keep"] == 3
    # the input table is not modified
    assert "LogFC" not in table.columns


def test_existing_logfc_is_kept():
    daa, _ = make_daa()
    table = pd.DataFrame(
        {"Mean(A)": [3.0], "Mean(B)": [1.0], "LogFC": [42.0]}, index=["g0"]
    )
    result = daa.compute_differential_attribution(
        table, "a", "RNA", A_IDX, B_IDX, ["a"], n_steps=1
    )
    assert result.loc["g0", "LogFC"] == 42.0


def test_compute_reference_batch_weights_distinct_batch_effects():
    # batch effects 0.25 (1 cell) and 9 (4 cells)
    mdata = make_mdata(X, batch=[9, 0.25, 9, 9, 9])
    daa, _ = make_daa(mdata=mdata)
    reference, weights, libsize = daa.compute_reference_batch("a", batch_size=2)
    np.testing.assert_allclose(
        np.asarray(reference[f"RNA_{Constants.TENSOR_NAME_BATCH}"]).ravel(), [0.25, 9]
    )
    np.testing.assert_allclose(weights, [0.2, 0.8])
    assert libsize == {}  # the fake decoder doesn't scale by the library size


def test_expected_values_average_over_the_batch_effects_of_all_cells():
    # regression (T15): the decoder took the batch effect of the first cell
    mdata = make_mdata(X, batch=[0.25, 9, 9, 9, 9])
    daa, _ = make_daa(mdata=mdata)
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=1
    )
    _, _, _, s0 = expected_terms()
    np.testing.assert_allclose(
        result["Expected(B)"], s0 + 0.2 * 0.25 + 0.8 * 9, atol=1e-5
    )


def test_attribution_decodes_at_the_median_libsize():
    # library sizes 6, 8, 3, 2, 4 -> median 4; batch effect 0
    model = FakeModel(E, W, libsize_scaling=True)
    daa = DifferentialAttributionAnalysis(make_mdata(X), model)
    result = daa.compute_differential_attribution(
        deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=3
    )
    zA, zB, d, s0 = expected_terms()
    np.testing.assert_allclose(result["Expected(B)"], 4 * s0, rtol=1e-5)
    np.testing.assert_allclose(
        result["Expected(A)"], 4 * (s0 + d["a"] + d["b"]), rtol=1e-5
    )
    # linear model: the attributions are exact at libsize 4
    np.testing.assert_allclose(result["attr_a"], 4 * d["a"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result["attr_b"], 4 * d["b"], rtol=1e-5, atol=1e-5)
    for libsize in model.decode_libsizes:
        np.testing.assert_allclose(np.asarray(libsize["RNA"]), 4.0)


# --------------------------------------------------------------------------
# from_deg_file
# --------------------------------------------------------------------------


@pytest.fixture
def read_csv(monkeypatch):
    m = MagicMock(name="read_csv")
    monkeypatch.setattr(f"{DAA_MODULE}.pd.read_csv", m)
    return m


SIGNIFICANT = Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANT_COLUMN


def deg_file_table(k, significant):
    return pd.DataFrame(
        {"K(A>B|Z)": k, SIGNIFICANT: significant}, index=FEATURES[: len(k)]
    )


def test_from_deg_file_keeps_significant_and_delegates(read_csv, monkeypatch):
    # the DEGs are the rows the differential analysis marked Significant;
    # the K values themselves are not re-thresholded here
    read_csv.return_value = deg_file_table([5.0, 9.0, -4.0], [True, False, True])
    daa, _ = make_daa()
    spy = MagicMock(return_value="result")
    monkeypatch.setattr(daa, "compute_differential_attribution", spy)

    out = daa.from_deg_file(
        "/in/deg.tsv", "a", "RNA", A_IDX, B_IDX, ["a"], batch_size=8, n_steps=5
    )
    assert out == "result"
    read_csv.assert_called_once_with(
        "/in/deg.tsv", sep=None, engine="python", index_col=0
    )
    kwargs = spy.call_args.kwargs
    assert list(kwargs["deg"].index) == ["g0", "g2"]
    assert kwargs["component"] == "a" and kwargs["modality"] == "RNA"
    assert kwargs["group_a_index"] == A_IDX and kwargs["group_b_index"] == B_IDX
    assert kwargs["with_respect_to"] == ["a"]
    assert kwargs["batch_size"] == 8 and kwargs["n_steps"] == 5


def test_from_deg_file_without_significant_column_skips(read_csv, monkeypatch, capsys):
    read_csv.return_value = pd.DataFrame({"K(A>B|Z)": [5.0]}, index=["g0"])
    daa, _ = make_daa()
    spy = MagicMock()
    monkeypatch.setattr(daa, "compute_differential_attribution", spy)
    assert daa.from_deg_file("f", "a", "RNA", A_IDX, B_IDX, ["a"]) is None
    spy.assert_not_called()
    assert "has no Significant column" in capsys.readouterr().out


def test_from_deg_file_too_few_degs_returns_none(read_csv, monkeypatch, capsys):
    read_csv.return_value = deg_file_table([5.0, 0.0], [True, False])
    daa, _ = make_daa()
    spy = MagicMock()
    monkeypatch.setattr(daa, "compute_differential_attribution", spy)
    assert (
        daa.from_deg_file("f", "a", "RNA", A_IDX, B_IDX, ["a"], min_num_degs=2) is None
    )
    spy.assert_not_called()
    assert "only 1 DEGs (min_num_degs=2)" in capsys.readouterr().out


def test_from_deg_file_end_to_end(read_csv):
    read_csv.return_value = deg_file_table([5.0, 0.0, 4.0], [True, False, True])
    daa, _ = make_daa()
    result = daa.from_deg_file("f", "a", "RNA", A_IDX, B_IDX, ["a", "b"], n_steps=2)
    _, _, d, _ = expected_terms()
    assert list(result.index) == ["g0", "g2"]
    np.testing.assert_allclose(result["attr_a"], d["a"][[0, 2]], atol=1e-5)


# ---------------------------------------------------------------------------
# open bugs found by the 2026-11-21 review
# ---------------------------------------------------------------------------


# regression: T15 the decoder's batch effect and library size came from the
# first cell of the whole dataset, so the result depended on the cell order
def test_differential_attribution_does_not_depend_on_cell_order():
    batch = [0.0, 1.0, 2.0, 3.0, 4.0]
    mdata = make_mdata(X, batch=batch)
    reordered = mdata[list(reversed(mdata.obs_names))].copy()
    results = [
        make_daa(mdata=m)[0].compute_differential_attribution(
            deg(), "a", "RNA", A_IDX, B_IDX, ["a", "b"], batch_size=2, n_steps=3
        )
        for m in (mdata, reordered)
    ]
    pd.testing.assert_frame_equal(results[0], results[1])
