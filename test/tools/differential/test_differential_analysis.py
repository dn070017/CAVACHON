"""Unit tests for cavachon.tools.differential_analysis.DifferentialAnalysis.

DataLoader / ReflectionHandler are fakes (see conftest). The model is a
``FakeModel`` whose decoded mean equals the RNA counts of the cell plus a
batch-effect term, so expected statistics can be computed by hand.
"""

from collections.abc import Mapping
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from test.tools.differential.conftest import (
    BATCH_KEY,
    X_KEY,
    FakeDataLoader,
    FakeModel,
    identity_model,
    make_mdata,
)

EPS = 1e-7
K_MAX = np.log(1.0 + EPS) - np.log(EPS)  # K when P == 1
K_MIN = np.log(EPS) - np.log(1.0 + EPS)  # K when P == 0
GROUP_A = Constants.DIFFERENTIAL_ANALYSIS_GROUP_A_COLUMN
GROUP_B = Constants.DIFFERENTIAL_ANALYSIS_GROUP_B_COLUMN
COMPONENT = Constants.DIFFERENTIAL_ANALYSIS_COMPONENT_COLUMN
MODALITY = Constants.DIFFERENTIAL_ANALYSIS_MODALITY_COLUMN
USE_CLUSTER = Constants.DIFFERENTIAL_ANALYSIS_USE_CLUSTER_COLUMN
SIGNIFICANT = Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANT_COLUMN
STATS = ["Mean(A)", "Mean(B)", "P(A>B|Z)", "P(B>A|Z)", "K(A>B|Z)", "K(B>A|Z)"]
COLUMNS = STATS + [SIGNIFICANT]
# the run's parameters, the same columns as in the HDEG tables
RUN_COLUMNS = [
    COMPONENT,
    MODALITY,
    "ZSamplingSize",
    "XSamplingSize",
    "RandomSeed",
    "SignificanceThreshold",
]


def make_analysis(mdata, model=None, **kwargs):
    kwargs.setdefault("batch_effect_colnames", {"RNA": []})
    kwargs.setdefault("distribution_names", {"RNA": "FakeDist"})
    model = model or identity_model(mdata["RNA"].n_vars)
    return DifferentialAnalysis(mdata, model, **kwargs)


def two_group_mdata():
    # cells 0-2: group "a" = [5, 0, 1]; cells 3-5: group "b" = [0, 5, 1]
    X = [[5, 0, 1]] * 3 + [[0, 5, 1]] * 3
    return make_mdata(X, clusters=["a"] * 3 + ["b"] * 3)


# --------------------------------------------------------------------------
# constructor
# --------------------------------------------------------------------------


def test_init_stores_arguments():
    mdata = two_group_mdata()
    model = MagicMock()
    encoders = {"RNA": {"batch": "enc"}}
    da = DifferentialAnalysis(mdata, model, {"RNA": ["b"]}, {"RNA": "d"}, encoders)
    assert da.mdata is mdata
    assert da.model is model
    assert da.batch_effect_colnames == {"RNA": ["b"]}
    assert da.distribution_names == {"RNA": "d"}
    assert da.batch_effect_encoders is encoders


def test_init_none_options_become_empty_mappings():
    # one place turns the None defaults into the mappings DataLoader takes,
    # so the subclasses and every DataLoader call can pass them as they are
    da = DifferentialAnalysis(two_group_mdata(), MagicMock())
    assert da.batch_effect_colnames == {}
    assert da.distribution_names == {}
    assert da.batch_effect_encoders == {}
    # a fresh dict per instance, not a shared default
    other = DifferentialAnalysis(two_group_mdata(), MagicMock())
    assert other.batch_effect_encoders is not da.batch_effect_encoders


# --------------------------------------------------------------------------
# compute_bayesian_factor
# --------------------------------------------------------------------------


def test_compute_bayesian_factor_hand_computed():
    a = np.array([[1, 0], [2, 0], [0, 0], [3, 5]], dtype=float)
    b = np.array([[0, 1], [0, 1], [1, 1], [0, 1]], dtype=float)
    da = make_analysis(two_group_mdata())
    result = da.compute_bayesian_factor(a, b, pd.Index(["x", "y"]))

    assert list(result.columns) == STATS
    assert list(result.index) == ["x", "y"]
    np.testing.assert_allclose(result["Mean(A)"], [1.5, 1.25])
    np.testing.assert_allclose(result["Mean(B)"], [0.25, 1.0])
    np.testing.assert_allclose(result["P(A>B|Z)"], [0.75, 0.25])
    np.testing.assert_allclose(result["P(B>A|Z)"], [0.25, 0.75])
    k_hi = np.log(0.75 + EPS) - np.log(0.25 + EPS)
    np.testing.assert_allclose(result["K(A>B|Z)"], [k_hi, -k_hi])
    np.testing.assert_allclose(result["K(B>A|Z)"], [-k_hi, k_hi])


def test_compute_bayesian_factor_extremes_are_finite():
    a = np.array([[2.0], [3.0]])
    b = np.array([[1.0], [1.0]])
    result = make_analysis(two_group_mdata()).compute_bayesian_factor(a, b, ["g"])
    assert result["K(A>B|Z)"].iloc[0] == pytest.approx(K_MAX)
    assert result["K(B>A|Z)"].iloc[0] == pytest.approx(K_MIN)
    assert np.isfinite(result.to_numpy()).all()


def test_compute_bayesian_factor_ties_count_for_neither_side():
    a = np.array([[1.0], [1.0]])
    result = make_analysis(two_group_mdata()).compute_bayesian_factor(a, a, ["g"])
    assert result["P(A>B|Z)"].iloc[0] == 0.0
    assert result["P(B>A|Z)"].iloc[0] == 0.0


def test_compute_bayesian_factor_nan_compares_false():
    a = np.array([[np.nan], [2.0]])
    b = np.array([[1.0], [1.0]])
    result = make_analysis(two_group_mdata()).compute_bayesian_factor(a, b, ["g"])
    assert result["P(A>B|Z)"].iloc[0] == 0.5
    assert np.isnan(result["Mean(A)"].iloc[0])


def test_compute_bayesian_factor_index_length_mismatch_raises():
    a = np.zeros((2, 2))
    with pytest.raises(ValueError):
        make_analysis(two_group_mdata()).compute_bayesian_factor(a, a, ["only_one"])


# --------------------------------------------------------------------------
# sample_mdata_x
# --------------------------------------------------------------------------


def test_sample_mdata_x_draws_from_index_and_keeps_modalities_aligned():
    n = 5
    X = np.arange(n, dtype=float).reshape(-1, 1) * np.ones((1, 2))
    atac = np.arange(n, dtype=float).reshape(-1, 1) * 10
    mdata = make_mdata(X, atac=atac)
    da = make_analysis(mdata)
    index = pd.Index(["cell1", "cell3"])

    sampled = da.sample_mdata_x(index, x_sampling_size=40)

    assert set(sampled.mod) == {"RNA", "ATAC"}
    rna, atac_s = sampled["RNA"], sampled["ATAC"]
    assert rna.n_obs == atac_s.n_obs == 40
    assert rna.obs_names.is_unique
    assert set(np.unique(rna.X[:, 0])) <= {1.0, 3.0}
    # the same original cell in the same row of every modality
    np.testing.assert_array_equal(atac_s.X[:, 0], rna.X[:, 0] * 10)
    np.testing.assert_array_equal(rna.obs_names, atac_s.obs_names)


def test_sample_mdata_x_accepts_list_and_is_seed_deterministic():
    mdata = make_mdata(np.arange(8, dtype=float).reshape(4, 2))
    da = make_analysis(mdata)
    np.random.seed(1)
    first = da.sample_mdata_x(["cell0", "cell2"], x_sampling_size=6)["RNA"].X.copy()
    np.random.seed(1)
    second = da.sample_mdata_x(["cell0", "cell2"], x_sampling_size=6)["RNA"].X
    np.testing.assert_array_equal(first, second)


def test_sample_mdata_x_empty_index_raises():
    da = make_analysis(two_group_mdata())
    with pytest.raises(ValueError):
        da.sample_mdata_x(pd.Index([]), x_sampling_size=3)


# --------------------------------------------------------------------------
# compute_x_means
# --------------------------------------------------------------------------


def dataset_from(x, be):
    return tf.data.Dataset.from_tensor_slices(
        {
            f"RNA_{X_KEY}": np.asarray(x, dtype=np.float32),
            f"RNA_{BATCH_KEY}": np.asarray(be, dtype=np.float32),
        }
    )


def test_compute_x_means_decodes_with_random_global_batch_effect(reflection):
    # 4 cells in the whole mdata; batch effect of global cell i is 100 * i
    mdata = make_mdata(np.zeros((4, 2)))
    model = identity_model(2)
    da = make_analysis(mdata, model)
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    dataset = dataset_from(x, np.zeros((3, 1)))
    batch_effect = {"RNA": tf.constant([[0.0], [100.0], [200.0], [300.0]])}

    np.random.seed(3)
    result = da.compute_x_means(dataset, "comp", "RNA", batch_effect, batch_size=2)

    np.random.seed(3)
    idx_1 = np.random.choice(np.arange(4), 2)
    idx_2 = np.random.choice(np.arange(4), 1)
    expected_be = 100.0 * np.concatenate([idx_1, idx_2]).reshape(-1, 1)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, x + expected_be)

    reflection.get_class_by_name.assert_called_once_with(
        "FakeDist", "distributions", "Distribution"
    )
    assert model.decode_calls == [(["comp"], False), (["comp"], False)]
    assert model.encode_calls == [False, False]


def test_compute_x_means_forwards_training_flag():
    mdata = make_mdata(np.zeros((2, 2)))
    model = identity_model(2)
    da = make_analysis(mdata, model)
    dataset = dataset_from(np.ones((2, 2)), np.zeros((2, 1)))
    da.compute_x_means(dataset, "comp", "RNA", {"RNA": tf.zeros((2, 1))}, training=True)
    assert model.encode_calls == [True]
    assert model.decode_calls == [(["comp"], True)]


def test_compute_x_means_unknown_component_raises():
    da = make_analysis(make_mdata(np.zeros((2, 2))))
    dataset = dataset_from(np.ones((2, 2)), np.zeros((2, 1)))
    with pytest.raises(AttributeError):
        da.compute_x_means(dataset, "missing", "RNA", {"RNA": tf.zeros((2, 1))})


# --------------------------------------------------------------------------
# between_two_groups
# --------------------------------------------------------------------------


def test_between_two_groups_end_to_end_statistics():
    mdata = two_group_mdata()
    da = make_analysis(mdata)
    result = da.between_two_groups(
        ["cell0", "cell1", "cell2"],
        ["cell3", "cell4", "cell5"],
        component="comp",
        modality="RNA",
        z_sampling_size=2,
        x_sampling_size=5,
        batch_size=4,
        sort_output=False,
    )
    assert list(result.columns) == COLUMNS + RUN_COLUMNS
    assert list(result.index) == ["g0", "g1", "g2"]
    # the run's parameters are recorded on every row
    assert (result[COMPONENT] == "comp").all()
    assert (result[MODALITY] == "RNA").all()
    assert (result["ZSamplingSize"] == 2).all()
    assert (result["XSamplingSize"] == 5).all()
    assert result["RandomSeed"].isna().all()
    assert (result["SignificanceThreshold"] == 3.2).all()
    # g2 is equal in both groups (both K ~ -16): not significant
    assert list(result[SIGNIFICANT]) == [True, True, False]
    np.testing.assert_allclose(result["Mean(A)"], [5, 0, 1])
    np.testing.assert_allclose(result["Mean(B)"], [0, 5, 1])
    np.testing.assert_allclose(result["P(A>B|Z)"], [1, 0, 0])
    np.testing.assert_allclose(result["P(B>A|Z)"], [0, 1, 0])
    np.testing.assert_allclose(result["K(A>B|Z)"], [K_MAX, K_MIN, K_MIN])


def test_between_two_groups_dataloader_arguments_and_sample_counts(monkeypatch):
    mdata = two_group_mdata()
    encoders = {"RNA": {}}
    da = make_analysis(mdata, batch_effect_encoders=encoders)
    calls = []
    real_sample = da.sample_mdata_x

    def spy(index, x_sampling_size):
        calls.append((list(index), x_sampling_size))
        return real_sample(index=index, x_sampling_size=x_sampling_size)

    monkeypatch.setattr(da, "sample_mdata_x", spy)
    da.between_two_groups(
        pd.Index(["cell0"]),
        pd.Index(["cell3"]),
        "comp",
        "RNA",
        z_sampling_size=3,
        x_sampling_size=7,
        batch_size=4,
    )
    # one loader over the whole mdata + two per z sample
    assert len(FakeDataLoader.instances) == 1 + 2 * 3
    first = FakeDataLoader.instances[0]
    assert first.mdata is mdata
    assert first.batch_size == 4
    assert first.batch_effect_colnames == {"RNA": []}
    assert first.distribution_names == {"RNA": "FakeDist"}
    assert first.batch_effect_encoders is encoders
    assert calls == [(["cell0"], 7), (["cell3"], 7)] * 3
    assert all(dl.mdata.n_obs == 7 for dl in FakeDataLoader.instances[1:])


def test_between_two_groups_stacks_all_z_samples(monkeypatch):
    da = make_analysis(two_group_mdata())
    draws = iter(
        [np.full((2, 3), v, dtype=float) for v in (1, 0, 2, 0, 3, 0)]
    )  # a, b, a, b, a, b
    monkeypatch.setattr(da, "compute_x_means", lambda **kw: next(draws))
    captured = {}
    real_bf = da.compute_bayesian_factor

    def spy(a, b, index):
        captured["a"], captured["b"] = a, b
        return real_bf(a, b, index)

    monkeypatch.setattr(da, "compute_bayesian_factor", spy)
    result = da.between_two_groups(
        ["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=3, x_sampling_size=2
    )
    assert captured["a"].shape == (6, 3)
    np.testing.assert_allclose(result["Mean(A)"], [2, 2, 2])
    np.testing.assert_allclose(result["Mean(B)"], [0, 0, 0])


def test_between_two_groups_sorts_by_larger_bayes_factor(monkeypatch):
    da = make_analysis(two_group_mdata())
    # P(A>B): g0 = 0.5 (|K| ~ 0), g1 = 1.0, g2 = 0.75
    a = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=float)
    b = np.full((4, 3), 0.5)
    draws = iter([a, b])
    monkeypatch.setattr(da, "compute_x_means", lambda **kw: next(draws))
    result = da.between_two_groups(
        ["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=1, x_sampling_size=4
    )
    assert list(result.index) == ["g1", "g2", "g0"]

    draws = iter([a, b])
    unsorted = da.between_two_groups(
        ["cell0"],
        ["cell3"],
        "comp",
        "RNA",
        z_sampling_size=1,
        x_sampling_size=4,
        sort_output=False,
    )
    assert list(unsorted.index) == ["g0", "g1", "g2"]


def test_between_two_groups_group_labels(monkeypatch):
    da = make_analysis(two_group_mdata())
    monkeypatch.setattr(da, "compute_x_means", lambda **kw: np.zeros((1, 3)))
    labelled = da.between_two_groups(
        ["cell0"],
        ["cell3"],
        "comp",
        "RNA",
        z_sampling_size=1,
        x_sampling_size=1,
        group_a_label="A1",
        group_b_label="B1",
        use_cluster="cluster",
    )
    assert (labelled[GROUP_A] == "A1").all()
    assert (labelled[GROUP_B] == "B1").all()
    assert (labelled[USE_CLUSTER] == "cluster").all()
    plain = da.between_two_groups(
        ["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=1, x_sampling_size=1
    )
    assert GROUP_A not in plain.columns and GROUP_B not in plain.columns
    assert USE_CLUSTER not in plain.columns
    assert list(plain.columns) == COLUMNS + RUN_COLUMNS
    assert (
        list(labelled.columns)
        == COLUMNS
        + [GROUP_A, GROUP_B]
        + RUN_COLUMNS[:2]
        + [USE_CLUSTER]
        + RUN_COLUMNS[2:]
    )


def test_between_two_groups_seed_is_reproducible_and_recorded():
    # random per-row noise so the random choices matter
    X = np.random.default_rng(0).random((8, 3))
    da = make_analysis(make_mdata(X, clusters=["a"] * 4 + ["b"] * 4))
    args = (["cell0", "cell1", "cell2", "cell3"], ["cell4", "cell5", "cell6", "cell7"])
    kwargs = dict(
        component="comp", modality="RNA", z_sampling_size=2, x_sampling_size=3
    )
    first = da.between_two_groups(*args, **kwargs, seed=7)
    np.random.seed(123)
    second = da.between_two_groups(*args, **kwargs, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["RandomSeed"] == 7).all()


def test_between_two_groups_zero_z_samples_raises():
    da = make_analysis(two_group_mdata())
    with pytest.raises(ValueError):
        da.between_two_groups(["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=0)


def test_between_two_groups_default_none_options_reach_dataloader_as_mappings():
    # regression: T5 - the None defaults reached DataLoader, which calls
    # .get() on them. Empty mappings make DataLoader read adata.uns.
    mdata = two_group_mdata()
    da = DifferentialAnalysis(mdata, identity_model(3))  # all defaults
    da.between_two_groups(
        ["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=1, x_sampling_size=2
    )
    for loader in FakeDataLoader.instances:
        assert isinstance(loader.batch_effect_colnames, Mapping)
        assert isinstance(loader.distribution_names, Mapping)


# --------------------------------------------------------------------------
# across_clusters_pairwise / across_clusters
# --------------------------------------------------------------------------


def test_bayes_factor_strength_is_signed():
    # g0: A>B, g1: B>A, g2: equal everywhere (both K ~ -16), g3: mostly
    # equal (both P = 0.02), g4: weak
    k_min = K_MIN
    k_weak = np.log(0.02) - np.log(0.98)
    table = k_table(
        [K_MAX, K_MIN, k_min, k_weak, 1.0], [K_MIN, K_MAX, k_min, k_weak, -1.0]
    )
    strength = DifferentialAnalysis.bayes_factor_strength(table)
    np.testing.assert_allclose(strength, [K_MAX, K_MAX, k_min, k_weak, 1.0])
    DifferentialAnalysis.mark_significant(table)
    assert list(table[SIGNIFICANT]) == [True, True, False, False, False]
    assert list(DifferentialAnalysis.keep_significant(table).index) == ["g0", "g1"]
    # ties sort last, not first as with |K|
    assert list(DifferentialAnalysis.sort_by_bayes_factor(table).index) == [
        "g0",
        "g1",
        "g4",
        "g3",
        "g2",
    ]


def test_mark_significant_threshold_is_inclusive():
    table = k_table([3.2, 3.19, -5.0], [0.0, 0.0, 3.2])
    assert list(table[SIGNIFICANT]) == [True, False, True]
    DifferentialAnalysis.mark_significant(table, threshold=5)
    assert not table[SIGNIFICANT].any()


@pytest.mark.parametrize(
    "values", [[True, False, True], ["True", "False", "True"]], ids=["bool", "str"]
)
def test_keep_significant_reads_the_column(values):
    # the column decides, not the K values; tables read back from disk may
    # hold strings
    table = k_table([0.0, 9.0, 0.0], [0.0, 0.0, 0.0])
    table[SIGNIFICANT] = values
    assert list(DifferentialAnalysis.keep_significant(table).index) == ["g0", "g2"]


def test_between_two_groups_significance_threshold(monkeypatch):
    da = make_analysis(two_group_mdata())
    # g0: P(A>B) = 1 (K ~ 16); g1: P(A>B) = 0.75 (K ~ 1.1); g2: P(B>A) = 1
    a = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=float)
    b = np.full((4, 3), 0.5)
    for threshold, expected in ((3.2, {"g0", "g2"}), (1.0, {"g0", "g1", "g2"})):
        draws = iter([a, b])
        monkeypatch.setattr(da, "compute_x_means", lambda **kw: next(draws))
        result = da.between_two_groups(
            ["cell0"],
            ["cell3"],
            "comp",
            "RNA",
            z_sampling_size=1,
            x_sampling_size=4,
            significance_threshold=threshold,
        )
        assert set(result.index[result[SIGNIFICANT]]) == expected
        assert (result["SignificanceThreshold"] == threshold).all()


def test_between_two_groups_identical_groups_sort_last(monkeypatch):
    # regression: T4 - sort_output ranked features with equal means first
    # (|K| ~ 16)
    da = make_analysis(two_group_mdata())
    a = np.array([[1.0, 2.0, 3.0]] * 4)
    b = np.array([[0.0, 2.0, 5.0]] * 4)  # g1 equal, g0 A>B, g2 B>A
    draws = iter([a, b])
    monkeypatch.setattr(da, "compute_x_means", lambda **kw: next(draws))
    result = da.between_two_groups(
        ["cell0"], ["cell3"], "comp", "RNA", z_sampling_size=1, x_sampling_size=4
    )
    assert list(result.index)[-1] == "g1"


def k_table(k_ab, k_ba):
    table = pd.DataFrame(
        {"K(A>B|Z)": k_ab, "K(B>A|Z)": k_ba}, index=[f"g{i}" for i in range(len(k_ab))]
    )
    DifferentialAnalysis.mark_significant(table)
    return table


def three_cluster_mdata():
    return make_mdata(np.zeros((6, 3)), clusters=["a", "a", "b", "c", "c", "c"])


def test_across_clusters_pairwise_calls_every_pair(monkeypatch):
    da = make_analysis(three_cluster_mdata())
    spy = MagicMock(side_effect=lambda **kw: k_table([0.0], [0.0]))
    monkeypatch.setattr(da, "between_two_groups", spy)

    results = da.across_clusters_pairwise(
        "comp",
        "RNA",
        "cluster",
        z_sampling_size=2,
        x_sampling_size=9,
        batch_size=8,
        sort_output=False,
    )
    assert list(results) == ["a/b", "a/c", "b/c"]
    first = spy.call_args_list[0].kwargs
    assert list(first["group_a_index"]) == ["cell0", "cell1"]
    assert list(first["group_b_index"]) == ["cell2"]
    assert first["group_a_label"] == "a" and first["group_b_label"] == "b"
    assert first["z_sampling_size"] == 2 and first["x_sampling_size"] == 9
    assert first["batch_size"] == 8 and first["sort_output"] is False
    assert first["component"] == "comp" and first["modality"] == "RNA"
    assert first["use_cluster"] == "cluster" and first["seed"] is None
    last = spy.call_args_list[2].kwargs
    assert list(last["group_b_index"]) == ["cell3", "cell4", "cell5"]


def test_across_clusters_pairwise_keep_only_significant(monkeypatch):
    da = make_analysis(three_cluster_mdata())
    table = k_table([5.0, 0.1, 3.2, 0.0], [0.0, 0.2, 0.0, 3.19])
    monkeypatch.setattr(da, "between_two_groups", lambda **kw: table.copy())
    results = da.across_clusters_pairwise(
        "comp", "RNA", "cluster", keep_only_significant=True
    )
    assert list(results["a/b"].index) == ["g0", "g2"]
    unfiltered = da.across_clusters_pairwise("comp", "RNA", "cluster")
    assert len(unfiltered["a/b"]) == 4


def test_across_clusters_pairwise_single_cluster_is_empty(monkeypatch):
    da = make_analysis(make_mdata(np.zeros((3, 2)), clusters=["a"] * 3))
    spy = MagicMock()
    monkeypatch.setattr(da, "between_two_groups", spy)
    assert da.across_clusters_pairwise("comp", "RNA", "cluster") == {}
    spy.assert_not_called()


def test_across_clusters_pairwise_missing_column_raises():
    da = make_analysis(three_cluster_mdata())
    with pytest.raises(KeyError):
        da.across_clusters_pairwise("comp", "RNA", "nope")


def test_across_clusters_one_vs_rest(monkeypatch):
    da = make_analysis(three_cluster_mdata())
    spy = MagicMock(side_effect=lambda **kw: k_table([0.0], [0.0]))
    monkeypatch.setattr(da, "between_two_groups", spy)
    results = da.across_clusters("comp", "RNA", "cluster", batch_size=16)

    assert list(results) == ["a", "b", "c"]
    calls = [c.kwargs for c in spy.call_args_list]
    assert list(calls[1]["group_a_index"]) == ["cell2"]
    assert list(calls[1]["group_b_index"]) == [
        "cell0",
        "cell1",
        "cell3",
        "cell4",
        "cell5",
    ]
    assert all(c["group_b_label"] == "others" for c in calls)
    assert [c["group_a_label"] for c in calls] == ["a", "b", "c"]
    assert all(c["use_cluster"] == "cluster" for c in calls)
    assert all(c["seed"] is None for c in calls)
    assert calls[0]["batch_size"] == 16


def test_across_clusters_keep_only_significant(monkeypatch):
    da = make_analysis(three_cluster_mdata())
    table = k_table([4.0, 1.0], [0.0, -1.0])
    monkeypatch.setattr(da, "between_two_groups", lambda **kw: table.copy())
    results = da.across_clusters("comp", "RNA", "cluster", keep_only_significant=True)
    assert all(list(df.index) == ["g0"] for df in results.values())


def test_across_clusters_single_cluster_has_empty_rest_group():
    # one-vs-rest with a single cluster leaves "others" empty
    da = make_analysis(make_mdata(np.ones((3, 2)), clusters=["a"] * 3))
    with pytest.raises(ValueError):
        da.across_clusters("comp", "RNA", "cluster", z_sampling_size=1)


def test_keep_only_significant_drops_features_identical_in_both_groups():
    # regression: T4 - keep_only_significant used |K| >= 3.2. Identical
    # decoded means give P(A>B) = P(B>A) = 0, so both K ~ -16 (evidence of
    # no difference) and every feature was kept as significant.
    mdata = make_mdata(np.ones((4, 2)), clusters=["a", "a", "b", "b"])
    da = make_analysis(mdata)
    results = da.across_clusters_pairwise(
        "comp",
        "RNA",
        "cluster",
        z_sampling_size=1,
        x_sampling_size=3,
        keep_only_significant=True,
    )
    assert len(results["a/b"]) == 0


# ---------------------------------------------------------------------------
# open bugs found by the 2026-11-21 review
# ---------------------------------------------------------------------------


# regression: T13 decoded means were scaled by each sampled cell's own library
# size, so groups that differ only in sequencing depth looked differential
def test_between_two_groups_depth_only_difference_is_not_significant():
    # same latent (z = gene 0) and same decoded profile; group b is sequenced
    # 5x deeper (library size 10 vs 2)
    X = [[1.0, 1.0]] * 4 + [[1.0, 9.0]] * 4
    model = FakeModel(
        {"comp": np.array([[1.0], [0.0]])},
        {"comp": np.array([[1.0, 1.0]])},
        libsize_scaling=True,
    )
    da = make_analysis(make_mdata(X), model=model)
    result = da.between_two_groups(
        [f"cell{i}" for i in range(4)],
        [f"cell{i}" for i in range(4, 8)],
        "comp",
        "RNA",
        z_sampling_size=2,
        x_sampling_size=20,
        seed=0,
    )
    assert not result[Constants.DIFFERENTIAL_ANALYSIS_SIGNIFICANT_COLUMN].any()
    # both groups decoded at the median library size (2 and 10 -> 6)
    np.testing.assert_allclose(result["Mean(A)"], [6.0, 6.0])
    np.testing.assert_allclose(result["Mean(B)"], [6.0, 6.0])
    assert (result[Constants.DIFFERENTIAL_ANALYSIS_LIBSIZE_COLUMN] == 6.0).all()


def libsize_model(n_vars=2):
    eye = np.eye(n_vars, dtype=np.float32)
    return FakeModel({"comp": eye}, {"comp": eye}, libsize_scaling=True)


def test_compute_decoding_reference_batch_effect_and_median_libsize():
    # library sizes 1, 2, 3, 10, 20 over batches of 2 -> median 3
    X = [[1, 0], [1, 1], [3, 0], [5, 5], [20, 0]]
    mdata = make_mdata(X, batch=[0, 1, 2, 3, 4])
    da = make_analysis(mdata, model=libsize_model())
    batch_effect, libsize = da.compute_decoding_reference("comp", 2)
    np.testing.assert_allclose(np.asarray(batch_effect["RNA"]).ravel(), range(5))
    assert libsize == {"RNA": 3.0}


def test_compute_decoding_reference_without_libsize_scaling():
    da = make_analysis(make_mdata(np.ones((3, 2))))
    _, libsize = da.compute_decoding_reference("comp", 2)
    assert libsize == {}


def test_compute_x_means_decodes_at_the_given_libsize():
    model = libsize_model()
    da = make_analysis(make_mdata(np.ones((3, 2))), model=model)
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            f"RNA_{X_KEY}": np.array([[1, 2], [3, 4], [5, 6]], np.float32),
            f"RNA_{BATCH_KEY}": np.zeros((3, 1), np.float32),
        }
    )
    out = da.compute_x_means(
        dataset,
        "comp",
        "RNA",
        {"RNA": tf.zeros((3, 1))},
        batch_size=2,
        libsize={"RNA": 7.0},
    )
    np.testing.assert_allclose(out, 7 * np.array([[1, 2], [3, 4], [5, 6]]))
    assert [tuple(np.shape(ls["RNA"])) for ls in model.decode_libsizes] == [
        (2, 1),
        (1, 1),
    ]


def test_compute_x_means_uses_the_given_batch_index_across_batches():
    da = make_analysis(make_mdata(np.zeros((3, 2))))
    x = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        {f"RNA_{X_KEY}": x, f"RNA_{BATCH_KEY}": np.zeros((3, 1), np.float32)}
    )
    pool = {"RNA": tf.constant([[0.0], [10.0], [20.0], [30.0]])}
    out = da.compute_x_means(
        dataset, "comp", "RNA", pool, batch_size=2, batch_index=np.array([3, 0, 2])
    )
    # identity model: x + the batch effect of pool cells 3, 0, 2
    np.testing.assert_allclose(out, x + [[30], [0], [20]])


def test_sample_batch_index_draws_from_the_pool():
    pool = {"RNA": tf.zeros((4, 1)), "ATAC": tf.zeros((4, 1))}
    np.random.seed(1)
    index = DifferentialAnalysis.sample_batch_index(pool, 6)
    np.random.seed(1)
    np.testing.assert_array_equal(index, np.random.choice(4, 6))


def test_between_two_groups_decodes_each_pair_with_one_batch_effect():
    # same latent in both groups and a pure batch effect: every pair ties
    # (with independent draws the batch noise alone gave P(A>B|Z) > 0)
    X = [[1.0, 2.0]] * 6
    mdata = make_mdata(X, batch=[0, 5, 10, 0, 5, 10])
    result = make_analysis(mdata).between_two_groups(
        ["cell0", "cell1", "cell2"],
        ["cell3", "cell4", "cell5"],
        "comp",
        "RNA",
        z_sampling_size=2,
        x_sampling_size=10,
        seed=0,
    )
    np.testing.assert_allclose(result["P(A>B|Z)"], 0)
    np.testing.assert_allclose(result["P(B>A|Z)"], 0)


def test_between_two_groups_without_libsize_scaling_has_no_column():
    da = make_analysis(make_mdata(np.ones((4, 2))))
    result = da.between_two_groups(
        ["cell0", "cell1"],
        ["cell2", "cell3"],
        "comp",
        "RNA",
        z_sampling_size=1,
        x_sampling_size=2,
    )
    assert Constants.DIFFERENTIAL_ANALYSIS_LIBSIZE_COLUMN not in result


@pytest.mark.xfail(
    strict=True,
    reason="BUG: T18 a NaN cluster label becomes a group with an empty index: "
    "numpy's 'a cannot be empty'",
)
def test_across_clusters_pairwise_skips_missing_labels():
    mdata = make_mdata(np.zeros((5, 3)), clusters=["a", "a", "b", "b", np.nan])
    results = make_analysis(mdata).across_clusters_pairwise(
        "comp", "RNA", "cluster", z_sampling_size=1, x_sampling_size=2
    )
    assert list(results) == ["a/b"]
