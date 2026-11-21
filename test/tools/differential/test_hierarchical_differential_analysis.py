"""Unit tests for cavachon.tools.hierarchical_differential_analysis.

DataLoader / ReflectionHandler are fakes (see conftest). DifferentialAnalysis
is the real base class. The FakeModel has two components ``a`` and ``b``
with ``z[c] = X`` and a decoder that only reads ``z_hat["a"]`` (unless a
test says otherwise), so substituting ``a`` changes the output and
substituting ``b`` does not.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.tools.differential_analysis import DifferentialAnalysis
from cavachon.tools.hierarchical_differential_analysis import (
    HierarchicalDifferentialAnalysis,
)
from test.tools.differential.conftest import (
    BATCH_KEY,
    X_KEY,
    FakeDataLoader,
    FakeModel,
    identity_model,
    make_mdata,
)

DONOR_COL = Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_DONOR_CLUSTER_COLUMN
RECIPIENT_COL = Constants.HIERARCHICAL_DIFFERENTIAL_ANALYSIS_RECIPIENT_CLUSTER_COLUMN


def make_hda(mdata, model=None, **kwargs):
    model = model or identity_model(mdata["RNA"].n_vars, components=("a", "b"))
    return HierarchicalDifferentialAnalysis(mdata, model, **kwargs)


def donor_recipient_mdata(categorical=False, extra_categories=()):
    # cells 0-2 donor "d" = [5, 0, 1]; cells 3-4 recipient "r" = [0, 5, 1]
    X = [[5, 0, 1]] * 3 + [[0, 5, 1]] * 2
    clusters = ["d"] * 3 + ["r"] * 2
    if categorical:
        clusters = pd.Categorical(clusters, categories=["d", "r", *extra_categories])
    return make_mdata(X, clusters=clusters)


def dataset_from(x):
    x = np.asarray(x, dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices(
        {
            f"RNA_{X_KEY}": x,
            f"RNA_{BATCH_KEY}": np.zeros((x.shape[0], 1), dtype=np.float32),
        }
    )


def test_is_a_differential_analysis():
    assert issubclass(HierarchicalDifferentialAnalysis, DifferentialAnalysis)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def test_make_dataloader_replaces_none_options_with_empty_dicts():
    mdata = donor_recipient_mdata()
    hda = make_hda(mdata)
    hda.batch_effect_encoders = None
    loader = hda._make_dataloader(mdata, 7)
    assert loader.mdata is mdata
    assert loader.batch_size == 7
    assert loader.batch_effect_colnames == {}
    assert loader.distribution_names == {}
    assert loader.batch_effect_encoders == {}


def test_make_dataloader_forwards_options():
    mdata = donor_recipient_mdata()
    enc = {"RNA": {"x": 1}}
    hda = make_hda(
        mdata,
        batch_effect_colnames={"RNA": ["batch"]},
        distribution_names={"RNA": "D"},
        batch_effect_encoders=enc,
    )
    loader = hda._make_dataloader(mdata, 3)
    assert loader.batch_effect_colnames == {"RNA": ["batch"]}
    assert loader.distribution_names == {"RNA": "D"}
    assert loader.batch_effect_encoders is enc


def test_require_dataset():
    hda = make_hda(donor_recipient_mdata())
    with pytest.raises(ValueError, match="Failed to create dataset"):
        hda._require_dataset(MagicMock(dataset=None))
    sentinel = object()
    assert hda._require_dataset(MagicMock(dataset=sentinel)) is sentinel


def test_get_cluster_mdata_subsets_every_modality():
    X = np.arange(4, dtype=float).reshape(-1, 1) * np.ones((1, 2))
    mdata = make_mdata(X, clusters=["x", "y", "x", "y"], atac=X[:, :1] * 10)
    hda = make_hda(mdata)
    sub = hda._get_cluster_mdata("y", "cluster", "RNA")
    assert set(sub.mod) == {"RNA", "ATAC"}
    assert list(sub["RNA"].obs_names) == ["cell1", "cell3"]
    np.testing.assert_array_equal(sub["ATAC"].X[:, 0], [10, 30])


def test_get_cluster_mdata_unknown_cluster_is_empty():
    hda = make_hda(donor_recipient_mdata())
    assert hda._get_cluster_mdata("zzz", "cluster", "RNA")["RNA"].n_obs == 0


def test_normalize_donor_components():
    hda = make_hda(donor_recipient_mdata())
    assert hda._normalize_donor_components(None, "a") == ["a"]
    assert hda._normalize_donor_components(["b", "a", "b"], "a") == ["b", "a"]
    with pytest.raises(ValueError, match="must not be empty"):
        hda._normalize_donor_components([], "a")
    with pytest.raises(ValueError, match=r"Unknown donor component\(s\): \['x'\]"):
        hda._normalize_donor_components(["a", "x"], "a")


def test_encode_z_pool_stacks_batches_per_component():
    X = np.arange(10, dtype=float).reshape(5, 2)
    mdata = make_mdata(X)
    E = {"a": np.eye(2), "b": 2 * np.eye(2)}
    model = FakeModel(E, {"a": np.eye(2), "b": np.eye(2)})
    hda = make_hda(mdata, model)
    pools = hda._encode_z_pool(mdata, ["b"], batch_size=2)
    assert list(pools) == ["b"]
    np.testing.assert_allclose(pools["b"], 2 * X)
    assert len(model.encode_calls) == 3  # ceil(5 / 2) batches


def test_encode_z_pool_empty_mdata():
    hda = make_hda(donor_recipient_mdata())
    empty = hda._get_cluster_mdata("zzz", "cluster", "RNA")
    pools = hda._encode_z_pool(empty, ["a", "b"], batch_size=4)
    assert pools["a"].shape == (0, 0) and pools["b"].shape == (0, 0)


def test_build_batch_effect_concatenates_batches():
    mdata = make_mdata(np.zeros((5, 2)), batch=[0, 1, 2, 3, 4])
    hda = make_hda(mdata)
    be = hda._build_batch_effect(batch_size=2)
    np.testing.assert_allclose(np.asarray(be["RNA"]).ravel(), [0, 1, 2, 3, 4])


# --------------------------------------------------------------------------
# _compute_substituted_x_means
# --------------------------------------------------------------------------


def test_compute_substituted_x_means_uses_one_donor_cell_for_all_components():
    # both components decode: x = z_a + z_b; pools encode the donor id
    model = FakeModel(
        {"a": np.eye(2), "b": np.eye(2)}, {"a": np.eye(2), "b": np.eye(2)}
    )
    mdata = make_mdata(np.zeros((4, 2)))
    hda = make_hda(mdata, model)
    ids = np.arange(1, 6, dtype=np.float32).reshape(-1, 1)
    pools = {"a": ids * np.ones((1, 2)), "b": 10 * ids * np.ones((1, 2))}

    np.random.seed(5)
    out = hda._compute_substituted_x_means(
        dataset_from(np.ones((3, 2))),
        "a",
        "RNA",
        pools,
        {"RNA": tf.zeros((4, 1))},
        batch_size=3,
    )
    np.random.seed(5)
    expected_idx = np.random.choice(5, size=3, replace=True)
    np.testing.assert_allclose(out, 11 * (expected_idx + 1).reshape(-1, 1) * [1, 1])


def test_compute_substituted_x_means_keeps_non_donor_components():
    mdata = make_mdata(np.zeros((2, 2)))
    model = identity_model(2, components=("a", "b"))
    hda = make_hda(mdata, model)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    pools = {"b": np.full((1, 2), 9.0, dtype=np.float32)}

    out = hda._compute_substituted_x_means(
        dataset_from(x), "a", "RNA", pools, {"RNA": tf.zeros((2, 1))}
    )
    # decode only reads a -> unchanged recipient values
    np.testing.assert_allclose(out, x)
    substituted_call = model.hier_calls[-1]
    np.testing.assert_allclose(substituted_call["a"], x)
    np.testing.assert_allclose(substituted_call["b"], np.full((2, 2), 9.0))
    assert model.decode_calls == [(["a"], False)]


# --------------------------------------------------------------------------
# between_clusters
# --------------------------------------------------------------------------


def test_between_clusters_substitution_statistics():
    hda = make_hda(donor_recipient_mdata())
    result = hda.between_clusters(
        "d", "r", "a", "RNA", "cluster", n_samples=2, seed=3, sort_output=False
    )
    assert list(result.index) == ["g0", "g1", "g2"]
    np.testing.assert_allclose(result["Mean(A)"], [5, 0, 1])  # substituted
    np.testing.assert_allclose(result["Mean(B)"], [0, 5, 1])  # original
    np.testing.assert_allclose(result["P(A>B|Z)"], [1, 0, 0])
    np.testing.assert_allclose(result["P(B>A|Z)"], [0, 1, 0])
    np.testing.assert_allclose(result["MeanDelta(Substituted-Original)"], [5, -5, 0])
    assert (result["InterventionType"] == "donor_pool_z_substitution").all()
    assert (result[DONOR_COL] == "d").all()
    assert (result[RECIPIENT_COL] == "r").all()
    assert (result["SamplingStrategy"] == "donor_pool").all()
    assert (result["RandomSeed"] == 3).all()
    assert (result["DonorComponents"] == "a").all()
    # the run's parameters are recorded on every row
    assert (result[Constants.DIFFERENTIAL_ANALYSIS_COMPONENT_COLUMN] == "a").all()
    assert (result[Constants.DIFFERENTIAL_ANALYSIS_MODALITY_COLUMN] == "RNA").all()
    assert (
        result[Constants.DIFFERENTIAL_ANALYSIS_USE_CLUSTER_COLUMN] == "cluster"
    ).all()
    assert (result["NSamples"] == 2).all()


def test_between_clusters_substituting_non_decoded_component_is_noop():
    hda = make_hda(donor_recipient_mdata())
    result = hda.between_clusters(
        "d", "r", "a", "RNA", "cluster", n_samples=1, donor_components=["b"]
    )
    np.testing.assert_allclose(result["MeanDelta(Substituted-Original)"], 0)
    assert (result["DonorComponents"] == "b").all()


def test_between_clusters_donor_components_column_is_sorted():
    hda = make_hda(donor_recipient_mdata())
    result = hda.between_clusters(
        "d", "r", "a", "RNA", "cluster", n_samples=1, donor_components=["b", "a", "b"]
    )
    assert (result["DonorComponents"] == "a,b").all()


def test_between_clusters_sorts_by_max_abs_bayes_factor():
    hda = make_hda(donor_recipient_mdata())
    result = hda.between_clusters("d", "r", "a", "RNA", "cluster", n_samples=1)
    k = result[["K(A>B|Z)", "K(B>A|Z)"]].abs().max(axis=1).to_numpy()
    assert (np.diff(k) <= 0).all()


def test_between_clusters_seed_is_reproducible(monkeypatch):
    # random per-row noise so the random choices matter
    X = np.random.default_rng(0).random((8, 3))
    mdata = make_mdata(X, clusters=["d"] * 4 + ["r"] * 4)
    hda = make_hda(mdata)
    first = hda.between_clusters("d", "r", "a", "RNA", "cluster", n_samples=2, seed=7)
    np.random.seed(123)
    second = hda.between_clusters("d", "r", "a", "RNA", "cluster", n_samples=2, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_between_clusters_uses_recipient_size_per_sample(monkeypatch):
    hda = make_hda(donor_recipient_mdata())
    sizes = []
    real = hda.sample_mdata_x

    def spy(index, x_sampling_size):
        sizes.append((list(index), x_sampling_size))
        return real(index, x_sampling_size=x_sampling_size)

    monkeypatch.setattr(hda, "sample_mdata_x", spy)
    hda.between_clusters("d", "r", "a", "RNA", "cluster", n_samples=3, batch_size=5)
    assert sizes == [(["cell3", "cell4"], 2)] * 3
    assert all(dl.batch_size == 5 for dl in FakeDataLoader.instances)


@pytest.mark.parametrize(
    "donor, recipient, match",
    [("x", "r", "Unknown donor cluster"), ("d", "x", "Unknown recipient cluster")],
)
def test_between_clusters_unknown_cluster_raises(donor, recipient, match):
    hda = make_hda(donor_recipient_mdata())
    with pytest.raises(ValueError, match=match):
        hda.between_clusters(donor, recipient, "a", "RNA", "cluster")


def test_between_clusters_empty_recipient_category_raises():
    hda = make_hda(donor_recipient_mdata(categorical=True, extra_categories=("e",)))
    with pytest.raises(ValueError, match="Recipient cluster 'e' is empty"):
        hda.between_clusters("d", "e", "a", "RNA", "cluster")


def test_between_clusters_empty_donor_category_raises():
    hda = make_hda(donor_recipient_mdata(categorical=True, extra_categories=("e",)))
    with pytest.raises(ValueError, match="Donor cluster 'e' is empty"):
        hda.between_clusters("e", "r", "a", "RNA", "cluster")


def test_between_clusters_unknown_donor_component_raises():
    hda = make_hda(donor_recipient_mdata())
    with pytest.raises(ValueError, match="Unknown donor component"):
        hda.between_clusters("d", "r", "a", "RNA", "cluster", donor_components=["zzz"])


def test_between_clusters_missing_cluster_column_raises():
    hda = make_hda(donor_recipient_mdata())
    with pytest.raises(KeyError):
        hda.between_clusters("d", "r", "a", "RNA", "nope")


# --------------------------------------------------------------------------
# across_clusters_pairwise
# --------------------------------------------------------------------------


def test_across_clusters_pairwise_runs_both_directions(monkeypatch):
    mdata = make_mdata(np.zeros((3, 2)), clusters=["x", "y", "z"])
    hda = make_hda(mdata)
    spy = MagicMock(
        side_effect=lambda **kw: f"{kw['donor_cluster']}>{kw['recipient_cluster']}"
    )
    monkeypatch.setattr(hda, "between_clusters", spy)

    results = hda.across_clusters_pairwise(
        "a",
        "RNA",
        "cluster",
        donor_components=["b"],
        n_samples=4,
        seed=9,
        batch_size=16,
        sort_output=False,
    )
    assert list(results) == ["x->y", "y->x", "x->z", "z->x", "y->z", "z->y"]
    assert results["y->x"] == "y>x"
    kw = spy.call_args_list[0].kwargs
    assert kw == dict(
        donor_cluster="x",
        recipient_cluster="y",
        component="a",
        modality="RNA",
        use_cluster="cluster",
        donor_components=["b"],
        n_samples=4,
        seed=9,
        batch_size=16,
        desc="x->y",
        sort_output=False,
    )


def test_across_clusters_pairwise_single_cluster_is_empty():
    hda = make_hda(make_mdata(np.zeros((2, 2)), clusters=["x", "x"]))
    assert hda.across_clusters_pairwise("a", "RNA", "cluster") == {}


def test_across_clusters_pairwise_end_to_end():
    hda = make_hda(donor_recipient_mdata())
    results = hda.across_clusters_pairwise("a", "RNA", "cluster", n_samples=1, seed=0)
    np.testing.assert_allclose(
        results["d->r"].loc[["g0", "g1", "g2"], "MeanDelta(Substituted-Original)"],
        [5, -5, 0],
    )
    np.testing.assert_allclose(
        results["r->d"].loc[["g0", "g1", "g2"], "MeanDelta(Substituted-Original)"],
        [-5, 5, 0],
    )
