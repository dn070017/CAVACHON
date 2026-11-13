"""Unit tests for cavachon.dataloader.DataLoader.

TensorUtils, ReflectionHandler and the layers used by the modifiers are
mocked (see conftest.py); file I/O (muon.read, tf.data save/load,
os.makedirs) is mocked per test.
"""
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tensorflow as tf

from cavachon.dataloader import DataLoader
from cavachon.dataloader.modifiers import IndependentBernoulliDataModifier
from test.dataloader.conftest import (
    DL_MODULE,
    N_OBS,
    FakeBinarize,
    FakeToDense,
    collect,
    fitted_encoder,
    make_adata,
    make_mdata,
    make_obs,
)


def batch_effect_side_effect(tensor_fn, new_encoders):
    """Side effect obeying create_tensor_from_df's contract: returns
    (tensor, encoder_dict) where encoder_dict is the passed-in dict updated
    with ``setdefault`` for each new encoder."""

    def side_effect(df, colnames, encoder_dict):
        for key, value in new_encoders.items():
            encoder_dict.setdefault(key, value)
        return tensor_fn(df), encoder_dict

    return side_effect


def one_hot_batch(df):
    return tf.constant(
        np.stack([(df["batch"] == c).to_numpy() for c in ["a", "b"]], 1).astype(np.float32)
    )


# --------------------------------------------------------------------------
# construction / dataset structure
# --------------------------------------------------------------------------


def test_dataset_keys_and_element_spec(mdata_two):
    dl = DataLoader(mdata_two, batch_size=4, batch_effect_encoders={})
    spec = dl.dataset.element_spec
    assert set(spec) == {"rna_matrix", "rna_batch_effect", "atac_matrix", "atac_batch_effect"}
    assert isinstance(spec["rna_matrix"], tf.SparseTensorSpec)  # no modifier configured
    assert spec["rna_matrix"].shape.as_list() == [5]
    assert spec["atac_matrix"].shape.as_list() == [4]
    assert spec["rna_batch_effect"].shape.as_list() == [1]
    assert spec["rna_batch_effect"].dtype == tf.float32


def test_spmatrix_converter_called_with_each_modality_X(mdata_two, mock_tensor_utils):
    DataLoader(mdata_two, batch_effect_encoders={})
    calls = mock_tensor_utils.spmatrix_to_sparse_tensor.call_args_list
    assert [c.args[0] for c in calls] == [mdata_two[m].X for m in mdata_two.mod]


def test_matrix_is_converter_return_value(mdata_two, mock_tensor_utils):
    sentinel = np.full((N_OBS, 2), 3.0, np.float32)
    mock_tensor_utils.spmatrix_to_sparse_tensor.side_effect = lambda X: tf.sparse.from_dense(
        sentinel
    )
    dl = DataLoader(mdata_two, batch_size=4, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), sentinel)
    np.testing.assert_array_equal(collect(dl, "atac_matrix"), sentinel)


def test_attributes_set_by_constructor(mdata_two, mock_tensor_utils):
    dl = DataLoader(mdata_two, batch_size=7, batch_effect_encoders={})
    assert dl.batch_size == 7
    assert dl.mdata is mdata_two
    assert dl.n_vars_batch_effect == {"rna": 1, "atac": 1}
    assert dl.batch_effect_encoder == {}
    # no batch-effect columns configured -> utils never consulted
    mock_tensor_utils.create_tensor_from_df.assert_not_called()


def test_cardinality_equals_n_obs(mdata_two):
    dl = DataLoader(mdata_two, batch_effect_encoders={})
    assert int(dl.dataset.cardinality()) == N_OBS


@pytest.mark.parametrize("batch_size,expected", [(3, [3, 3, 3, 1]), (5, [5, 5]), (128, [10])])
def test_iter_batches_sizes(mdata_two, batch_size, expected):
    dl = DataLoader(mdata_two, batch_size=batch_size, batch_effect_encoders={})
    assert [int(b["rna_batch_effect"].shape[0]) for b in dl] == expected


def test_iter_is_reiterable_and_deterministic(mdata_two):
    dl = DataLoader(mdata_two, batch_size=3, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), collect(dl, "rna_matrix"))


def test_no_shuffling_order_matches_obs(mdata_two):
    """Row i of the dataset must be cell i of the modality (embedding alignment)."""
    dl = DataLoader(mdata_two, batch_size=3, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), mdata_two["rna"].X.toarray())
    np.testing.assert_array_equal(collect(dl, "atac_matrix"), mdata_two["atac"].X.toarray())


def test_order_preserved_with_non_sorted_obs_names(sparse_counts):
    perm = np.array([7, 2, 9, 0, 5, 1, 8, 3, 6, 4])
    adata = make_adata(sp.csr_matrix(sparse_counts.toarray()[perm]), obs=make_obs().iloc[perm])
    dl = DataLoader(make_mdata({"rna": adata}), batch_size=4, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), sparse_counts.toarray()[perm])


def test_unequal_n_obs_across_modalities_raises(sparse_counts):
    rna = make_adata(sparse_counts)
    atac = make_adata(sp.csr_matrix(np.ones((6, 3), np.float32)), obs=make_obs(6, prefix="x"))
    with pytest.raises(ValueError, match="obs_names"):
        DataLoader(make_mdata({"rna": rna, "atac": atac}), batch_effect_encoders={})


def test_permuted_obs_across_modalities_raises():
    # regression (H3): modalities with the same cells in a different
    # order used to be paired by row position, silently misaligning
    # cells. Model.predict writes results back to each modality by row
    # position, so the DataLoader rejects misaligned input instead of
    # reordering it.
    obs = make_obs()
    ids = np.arange(N_OBS, dtype=np.float32).reshape(-1, 1) + 1
    perm = np.arange(N_OBS)[::-1]
    rna = make_adata(sp.csr_matrix(ids), obs=obs.copy())
    atac = make_adata(sp.csr_matrix(ids[perm]), obs=obs.iloc[perm].copy())
    with pytest.raises(ValueError, match="obs_names of modality 'atac'"):
        DataLoader(make_mdata({"rna": rna, "atac": atac}), batch_size=4, batch_effect_encoders={})


def test_rows_aligned_across_modalities_with_same_obs_order():
    obs = make_obs()
    ids = np.arange(N_OBS, dtype=np.float32).reshape(-1, 1) + 1
    rna = make_adata(sp.csr_matrix(ids), obs=obs.copy())
    atac = make_adata(sp.csr_matrix(ids * 10), obs=obs.copy())
    dl = DataLoader(make_mdata({"rna": rna, "atac": atac}), batch_size=4, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_matrix") * 10, collect(dl, "atac_matrix"))


def test_input_mdata_not_mutated(mdata_two):
    before_X = mdata_two["rna"].X.copy()
    before_obs = mdata_two["rna"].obs.copy()
    before_names = mdata_two.obs_names.copy()
    DataLoader(mdata_two, batch_effect_encoders={}, distribution_names={"rna": "IndependentBernoulli"})
    assert (mdata_two["rna"].X != before_X).nnz == 0
    pd.testing.assert_frame_equal(mdata_two["rna"].obs, before_obs)
    assert mdata_two.obs_names.equals(before_names)


# --------------------------------------------------------------------------
# batch effect covariates (DataLoader's wiring and bookkeeping)
# --------------------------------------------------------------------------


def test_batch_effect_args_forwarded_to_utils(sparse_counts, mock_tensor_utils):
    adata = make_adata(sparse_counts)
    encoders = {"rna": {}}
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": fitted_encoder("ab")}
    )
    DataLoader(make_mdata({"rna": adata}), batch_effect_colnames={"rna": ["batch"]},
               batch_effect_encoders=encoders)
    (df, colnames, enc), _ = mock_tensor_utils.create_tensor_from_df.call_args
    pd.testing.assert_frame_equal(df, adata.obs)
    assert colnames == ["batch"]
    assert enc is encoders["rna"]


def test_batch_effect_tensor_placed_in_dataset(sparse_counts, mock_tensor_utils):
    adata = make_adata(sparse_counts)
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": fitted_encoder("ab")}
    )
    dl = DataLoader(make_mdata({"rna": adata}), batch_size=4,
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})
    np.testing.assert_array_equal(collect(dl, "rna_batch_effect"), one_hot_batch(adata.obs).numpy())


def test_categorical_encoder_bookkeeping(sparse_counts, mock_tensor_utils):
    enc = fitted_encoder("ab")
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": enc}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})
    assert dl.n_vars_batch_effect == {"rna": 2}
    assert dl.batch_effect_encoder == {"rna_batch": enc}
    assert dl.batch_effect_encoders == {"rna": {"batch": enc}}


def test_continuous_encoder_bookkeeping(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.constant(df[["depth"]].to_numpy(np.float32)), {"depth": None}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["depth"]}, batch_effect_encoders={})
    assert dl.n_vars_batch_effect == {"rna": 1}
    assert dl.batch_effect_encoder == {"rna_depth": None}


def test_mixed_encoder_bookkeeping(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.zeros((len(df), 3)), {"batch": fitted_encoder("ab"), "depth": None}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch", "depth"]}, batch_effect_encoders={})
    assert dl.n_vars_batch_effect == {"rna": 3}
    assert dl.dataset.element_spec["rna_batch_effect"].shape.as_list() == [3]


def test_batch_effect_colnames_from_uns(sparse_counts, mock_tensor_utils):
    adata = make_adata(sparse_counts, batch_effect_colnames=["batch"])
    DataLoader(make_mdata({"rna": adata}), batch_effect_encoders={})
    assert mock_tensor_utils.create_tensor_from_df.call_args.args[1] == ["batch"]


def test_batch_effect_parameter_overrides_uns(sparse_counts, mock_tensor_utils):
    adata = make_adata(sparse_counts, batch_effect_colnames=["batch"])
    DataLoader(make_mdata({"rna": adata}), batch_effect_colnames={"rna": ["depth"]},
               batch_effect_encoders={})
    assert mock_tensor_utils.create_tensor_from_df.call_args.args[1] == ["depth"]


def test_empty_batch_effect_list_gives_zeros(sparse_counts, mock_tensor_utils):
    adata = make_adata(sparse_counts, batch_effect_colnames=["batch"])
    dl = DataLoader(make_mdata({"rna": adata}), batch_effect_colnames={"rna": []},
                    batch_effect_encoders={})
    mock_tensor_utils.create_tensor_from_df.assert_not_called()
    np.testing.assert_array_equal(collect(dl, "rna_batch_effect"), np.zeros((N_OBS, 1)))
    assert dl.n_vars_batch_effect == {"rna": 1}


def test_batch_effect_only_for_configured_modality(mdata_two, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": fitted_encoder("ab")}
    )
    dl = DataLoader(mdata_two, batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})
    assert mock_tensor_utils.create_tensor_from_df.call_count == 1
    assert dl.n_vars_batch_effect == {"rna": 2, "atac": 1}
    np.testing.assert_array_equal(collect(dl, "atac_batch_effect"), np.zeros((N_OBS, 1)))


def test_supplied_encoder_dict_is_populated_in_place(sparse_counts, mock_tensor_utils):
    enc = fitted_encoder("ab")
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": enc}
    )
    encoders = {}
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders=encoders)
    assert dl.batch_effect_encoders is encoders
    assert encoders["rna"]["batch"] is enc


def test_supplied_encoders_reused_by_second_loader(sparse_counts, mock_tensor_utils):
    enc = fitted_encoder("ab")
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": enc}
    )
    shared = {}
    adata = make_adata(sparse_counts)
    DataLoader(make_mdata({"rna": adata}), batch_effect_colnames={"rna": ["batch"]},
               batch_effect_encoders=shared)
    DataLoader(make_mdata({"rna": adata.copy()}), batch_effect_colnames={"rna": ["batch"]},
               batch_effect_encoders=shared)
    second_call_encoders = mock_tensor_utils.create_tensor_from_df.call_args_list[1].args[2]
    assert second_call_encoders["batch"] is enc


def test_utils_error_propagates(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = ValueError("unseen labels")
    with pytest.raises(ValueError, match="unseen labels"):
        DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                   batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})


def test_default_batch_effect_encoders_not_shared(sparse_counts, mock_tensor_utils):
    # regression (H2): the default batch_effect_encoders used to be a
    # shared mutable dict(), leaking encoders between DataLoaders
    side_effect = batch_effect_side_effect(one_hot_batch, {"batch": fitted_encoder("ab")})
    encoders_at_call = []

    def recording_side_effect(df, colnames, encoder_dict):
        encoders_at_call.append(dict(encoder_dict))
        return side_effect(df, colnames, encoder_dict)

    mock_tensor_utils.create_tensor_from_df.side_effect = recording_side_effect
    dl1 = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                     batch_effect_colnames={"rna": ["batch"]})
    dl2 = DataLoader(make_mdata({"rna": make_adata(sparse_counts.copy())}),
                     batch_effect_colnames={"rna": ["batch"]})
    # a fresh loader must start with no encoders of its own
    assert encoders_at_call == [{}, {}]
    assert dl1.batch_effect_encoders is not dl2.batch_effect_encoders


def test_passed_batch_effect_encoders_updated_inplace(sparse_counts, mock_tensor_utils):
    encoder = fitted_encoder("ab")
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": encoder}
    )
    encoders = {}
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders=encoders)
    assert dl.batch_effect_encoders is encoders
    assert encoders == {"rna": {"batch": encoder}}


# regression: n_vars_batch_effect was 0 for a width-1 tensor when no encoder was returned
def test_n_vars_matches_tensor_width_when_no_encoder_returned(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.zeros((len(df), 1)), {}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["does_not_exist"]}, batch_effect_encoders={})
    width = dl.dataset.element_spec["rna_batch_effect"].shape[-1]
    assert dl.n_vars_batch_effect["rna"] == width


# regression: stale encoders of unrequested columns were counted in n_vars_batch_effect
def test_n_vars_ignores_stale_encoders_for_unrequested_columns(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.constant(df[["depth"]].to_numpy(np.float32)), {"depth": None}
    )
    encoders = {"rna": {"batch": fitted_encoder("ab")}}  # from an earlier run
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["depth"]}, batch_effect_encoders=encoders)
    width = dl.dataset.element_spec["rna_batch_effect"].shape[-1]
    assert width == 1
    assert dl.n_vars_batch_effect["rna"] == width
    # only the requested column is tracked in the flat mapping; the reused
    # encoder is kept in the nested dict for later loaders
    assert dl.batch_effect_encoder == {"rna_depth": None}
    assert "batch" in dl.batch_effect_encoders["rna"]


# regression: n_vars_batch_effect used the encoder class count instead of the tensor width
def test_n_vars_matches_tensor_width_with_reused_categorical_encoder(
    sparse_counts, mock_tensor_utils
):
    enc = fitted_encoder([0, 1])
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.zeros((len(df), 1)), {}  # continuous column returned
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]},
                    batch_effect_encoders={"rna": {"batch": enc}})
    width = dl.dataset.element_spec["rna_batch_effect"].shape[-1]
    assert dl.n_vars_batch_effect["rna"] == width


# regression: calling create_dataset() again doubled n_vars_batch_effect
def test_create_dataset_idempotent_n_vars(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": fitted_encoder("ab")}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})
    dl.create_dataset({"rna": ["batch"]})
    assert dl.n_vars_batch_effect["rna"] == 2


def test_create_dataset_again_resets_bookkeeping(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        one_hot_batch, {"batch": fitted_encoder("ab")}
    )
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}),
                    batch_effect_colnames={"rna": ["batch"]}, batch_effect_encoders={})
    dl.create_dataset({"rna": []})
    assert dl.n_vars_batch_effect == {"rna": 1}
    assert dl.batch_effect_encoder == {}
    assert dl.dataset.element_spec["rna_batch_effect"].shape.as_list() == [1]


# --------------------------------------------------------------------------
# distribution modifiers via DataLoader (ReflectionHandler mocked; modifiers real;
# layers faked)
# --------------------------------------------------------------------------


def test_modifier_resolved_by_name_from_uns(sparse_counts, mock_reflection):
    adata = make_adata(sparse_counts, distribution="IndependentBernoulli")
    DataLoader(make_mdata({"atac": adata}), batch_effect_encoders={})
    mock_reflection.assert_called_once_with(
        "IndependentBernoulli", "dataloader/modifiers", "DataModifier"
    )


def test_distribution_param_overrides_uns(sparse_counts, mock_reflection):
    adata = make_adata(sparse_counts, distribution="IndependentBernoulli")
    DataLoader(make_mdata({"rna": adata}), distribution_names={"rna": "MultivariateNormalDiag"},
               batch_effect_encoders={})
    assert mock_reflection.call_args.args[0] == "MultivariateNormalDiag"


def test_no_distribution_no_lookup_and_stays_sparse(sparse_counts, mock_reflection):
    dl = DataLoader(make_mdata({"rna": make_adata(sparse_counts)}), batch_effect_encoders={})
    mock_reflection.assert_not_called()
    assert isinstance(dl.dataset.element_spec["rna_matrix"], tf.SparseTensorSpec)


def test_modifier_instantiated_with_modality_name(sparse_counts, mock_reflection):
    created = []

    class Spy(IndependentBernoulliDataModifier):
        def __init__(self, modality_name):
            created.append(modality_name)
            super().__init__(modality_name)

    mock_reflection.side_effect = lambda *a: Spy
    DataLoader(make_mdata({"atac": make_adata(sparse_counts, distribution="X")}),
               batch_effect_encoders={})
    assert created == ["atac"]


def test_bernoulli_applies_todense_then_binarize(fake_layers):
    X = np.array([[0, 1, 3], [2, 0, 0.5]] * (N_OBS // 2), dtype=np.float32)
    adata = make_adata(sp.csr_matrix(X), distribution="IndependentBernoulli")
    dl = DataLoader(make_mdata({"atac": adata}), batch_size=3, batch_effect_encoders={})
    fake_layers.ToDense.assert_called_once_with("atac_matrix")
    fake_layers.Binarize.assert_called_once_with("atac_matrix")
    assert isinstance(dl.dataset.element_spec["atac_matrix"], tf.TensorSpec)
    np.testing.assert_array_equal(collect(dl, "atac_matrix"), (X >= 1).astype(np.float32))


@pytest.mark.parametrize(
    "distribution",
    ["IndependentZeroInflatedNegativeBinomial", "MultivariateNormalDiag", "Studentt"],
)
def test_todense_distributions_preserve_values(counts, distribution, fake_layers):
    X = counts.copy()
    X[0, 0] = 2.5
    adata = make_adata(sp.csr_matrix(X), distribution=distribution)
    dl = DataLoader(make_mdata({"rna": adata}), batch_size=4, batch_effect_encoders={})
    fake_layers.ToDense.assert_called_once_with("rna_matrix")
    fake_layers.Binarize.assert_not_called()
    assert isinstance(dl.dataset.element_spec["rna_matrix"], tf.TensorSpec)
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), X)


def test_modifier_only_applied_to_its_modality(mdata_two):
    dl = DataLoader(mdata_two, distribution_names={"atac": "IndependentBernoulli"},
                    batch_effect_encoders={})
    spec = dl.dataset.element_spec
    assert isinstance(spec["atac_matrix"], tf.TensorSpec)
    assert isinstance(spec["rna_matrix"], tf.SparseTensorSpec)
    np.testing.assert_array_equal(collect(dl, "rna_matrix"), mdata_two["rna"].X.toarray())


def test_modifiers_do_not_touch_batch_effect(sparse_counts, mock_tensor_utils):
    mock_tensor_utils.create_tensor_from_df.side_effect = batch_effect_side_effect(
        lambda df: tf.constant(df[["depth"]].to_numpy(np.float32)), {"depth": None}
    )
    adata = make_adata(sparse_counts, distribution="IndependentBernoulli")
    dl = DataLoader(make_mdata({"rna": adata}), batch_effect_colnames={"rna": ["depth"]},
                    batch_effect_encoders={})
    np.testing.assert_allclose(collect(dl, "rna_batch_effect")[:, 0], adata.obs["depth"])


def test_modify_dataset_twice_maps_modifier_twice(sparse_counts):
    adata = make_adata(sparse_counts, distribution="IndependentBernoulli")
    dl = DataLoader(make_mdata({"atac": adata}), batch_effect_encoders={})
    dl.modify_dataset()
    # a second modifier instance is created and chained onto the dataset
    assert len(FakeToDense.instances) == 2 and len(FakeBinarize.instances) == 2
    np.testing.assert_array_equal(
        collect(dl, "atac_matrix"), (sparse_counts.toarray() >= 1).astype(np.float32)
    )


def test_unknown_distribution_error_propagates(sparse_counts):
    adata = make_adata(sparse_counts, distribution="NotADistribution")
    with pytest.raises(AssertionError):
        DataLoader(make_mdata({"rna": adata}), batch_effect_encoders={})


# --------------------------------------------------------------------------
# I/O (all file access mocked)
# --------------------------------------------------------------------------


def test_from_h5mu_reads_updates_and_builds(monkeypatch, mdata_two):
    read = mock.MagicMock(return_value=mdata_two)
    update = mock.MagicMock()
    monkeypatch.setattr(f"{DL_MODULE}.mu.read", read)
    monkeypatch.setattr(mdata_two, "update", update)
    dl = DataLoader.from_h5mu("some/dir/../data.h5mu", batch_size=3)
    read.assert_called_once_with(os.path.realpath("some/dir/../data.h5mu"))
    update.assert_called_once_with()
    assert dl.mdata is mdata_two
    assert dl.batch_size == 3
    assert set(dl.dataset.element_spec) == {
        "rna_matrix", "rna_batch_effect", "atac_matrix", "atac_batch_effect"
    }


def test_save_dataset_creates_dir_and_saves(monkeypatch, mdata_two):
    # uses tf.data.Dataset.save (tf.data.experimental.save is deprecated)
    makedirs = mock.MagicMock()
    monkeypatch.setattr(f"{DL_MODULE}.os.makedirs", makedirs)
    dl = DataLoader(mdata_two, batch_effect_encoders={})
    dl.dataset = mock.MagicMock(name="dataset")
    dl.save_dataset("out/../ds")
    path = os.path.realpath("out/../ds")
    makedirs.assert_called_once_with(path, exist_ok=True)
    dl.dataset.save.assert_called_once_with(path)


def test_load_dataset_replaces_dataset(monkeypatch, mdata_two):
    loaded = tf.data.Dataset.from_tensor_slices({"rna_matrix": tf.ones((2, 5))})
    load = mock.MagicMock(return_value=loaded)
    monkeypatch.setattr(f"{DL_MODULE}.tf.data.Dataset.load", load)
    dl = DataLoader(mdata_two, batch_effect_encoders={})
    dl.load_dataset("ds")
    load.assert_called_once_with(os.path.realpath("ds"))
    assert dl.dataset is loaded
