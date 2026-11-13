"""Shared fixtures and mocks for ``cavachon.dataloader`` unit tests.

Package boundary: ``cavachon.dataloader`` (DataLoader + dataloader/modifiers)
stays real. Everything imported from other cavachon packages is mocked:

* ``cavachon.utils.TensorUtils``      -> ``mock_tensor_utils`` (autouse)
* ``cavachon.utils.ReflectionHandler`` -> ``mock_reflection`` (autouse)
* ``cavachon.layers.modifiers.ToDense/Binarize`` -> ``FakeToDense`` /
  ``FakeBinarize`` patched into every modifier module (autouse)

``cavachon.environment.Constants`` is plain constants and is used as-is.
"""

import warnings
from unittest import mock

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from cavachon.dataloader.modifiers import (
    IndependentBernoulliDataModifier,
    IndependentZeroInflatedNegativeBinomialDataModifier,
    MultivariateNormalDiagDataModifier,
    StudenttDataModifier,
)

N_OBS = 10
DL_MODULE = "cavachon.dataloader.dataloader"

MODIFIER_MODULES = {
    "IndependentBernoulli": "cavachon.dataloader.modifiers.independent_bernoulli_data_modifier",
    "IndependentZeroInflatedNegativeBinomial": (
        "cavachon.dataloader.modifiers."
        "independent_zero_inflated_negative_binomial_data_modifier"
    ),
    "MultivariateNormalDiag": "cavachon.dataloader.modifiers.multivariate_normal_diag_data_modifier",
    "Studentt": "cavachon.dataloader.modifiers.studentt_data_modifier",
}
MODIFIER_CLASSES = {
    "IndependentBernoulli": IndependentBernoulliDataModifier,
    "IndependentZeroInflatedNegativeBinomial": IndependentZeroInflatedNegativeBinomialDataModifier,
    "MultivariateNormalDiag": MultivariateNormalDiagDataModifier,
    "Studentt": StudenttDataModifier,
}


# --------------------------------------------------------------------------
# synthetic data builders (third-party objects only)
# --------------------------------------------------------------------------


def make_obs(n_obs: int = N_OBS, prefix: str = "cell") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch": ["a", "b"] * (n_obs // 2) + ["a"] * (n_obs % 2),
            "depth": np.linspace(0.5, 7.5, n_obs),
        },
        index=[f"{prefix}{i}" for i in range(n_obs)],
    )


def make_adata(X, obs=None, distribution=None, batch_effect_colnames=None):
    obs = make_obs(X.shape[0]) if obs is None else obs
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f"g{j}" for j in range(X.shape[1])]
    config = {}
    if distribution is not None:
        config["distribution"] = distribution
    if batch_effect_colnames is not None:
        config["batch_effect_colnames"] = batch_effect_colnames
    if config:
        adata.uns["cavachon"] = config
    return adata


def make_mdata(mods):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mu.MuData(mods)


def fitted_encoder(classes):
    return LabelEncoder().fit(list(classes))


def collect(dataloader, key):
    """Concatenate a key across every batch of a DataLoader, densifying."""
    chunks = []
    for batch in dataloader:
        value = batch[key]
        if isinstance(value, tf.SparseTensor):
            value = tf.sparse.to_dense(value)
        chunks.append(value.numpy())
    return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------
# fakes for cavachon.layers.modifiers (plain classes, not MagicMock, because
# keras attribute tracking would probe MagicMock attributes)
# --------------------------------------------------------------------------


class FakeToDense:
    instances = []

    def __init__(self, key, *args, **kwargs):
        self.key = key
        self.calls = 0
        FakeToDense.instances.append(self)

    def __call__(self, inputs):
        self.calls += 1
        inputs = dict(inputs)
        if isinstance(inputs[self.key], tf.SparseTensor):
            inputs[self.key] = tf.sparse.to_dense(inputs[self.key])
        return inputs


class FakeBinarize:
    """Marks values >= 1 as 1 and everything else as 0 (fake contract)."""

    instances = []

    def __init__(self, key, threshold=1.0, *args, **kwargs):
        self.key = key
        self.threshold = threshold
        self.calls = 0
        FakeBinarize.instances.append(self)

    def __call__(self, inputs):
        self.calls += 1
        inputs = dict(inputs)
        t = inputs[self.key]
        inputs[self.key] = tf.cast(t >= self.threshold, t.dtype)
        return inputs


@pytest.fixture(autouse=True)
def fake_layers(monkeypatch):
    FakeToDense.instances = []
    FakeBinarize.instances = []
    to_dense = mock.MagicMock(side_effect=FakeToDense, name="ToDense")
    binarize = mock.MagicMock(side_effect=FakeBinarize, name="Binarize")
    for module in MODIFIER_MODULES.values():
        monkeypatch.setattr(f"{module}.ToDense", to_dense)
    monkeypatch.setattr(
        f"{MODIFIER_MODULES['IndependentBernoulli']}.Binarize", binarize
    )
    return mock.Mock(ToDense=to_dense, Binarize=binarize)


# --------------------------------------------------------------------------
# mocks for cavachon.utils (patched where looked up in dataloader.py)
# --------------------------------------------------------------------------


def fake_spmatrix_to_sparse_tensor(X):
    """Test-controlled return value: a real tf.SparseTensor of X (float32)."""
    coo = sp.coo_matrix(X)
    indices = np.stack([coo.row, coo.col], axis=1).astype(np.int64)
    st = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)
    return tf.sparse.reorder(st)


@pytest.fixture(autouse=True)
def mock_tensor_utils(monkeypatch):
    to_sparse = mock.MagicMock(side_effect=fake_spmatrix_to_sparse_tensor)
    # default: one continuous column -> width 1, encoder None. Tests override
    # return_value / side_effect to control the batch-effect tensor.
    create_df = mock.MagicMock(
        side_effect=lambda df, colnames, enc: (
            tf.zeros((len(df), 1)),
            enc,
        )
    )
    monkeypatch.setattr(f"{DL_MODULE}.TensorUtils.spmatrix_to_sparse_tensor", to_sparse)
    monkeypatch.setattr(f"{DL_MODULE}.TensorUtils.create_tensor_from_df", create_df)
    return mock.Mock(
        spmatrix_to_sparse_tensor=to_sparse, create_tensor_from_df=create_df
    )


@pytest.fixture(autouse=True)
def mock_reflection(monkeypatch):
    def resolve(name, subdirectory, postfix):
        if name not in MODIFIER_CLASSES:
            raise AssertionError(f"unknown {name}")
        return MODIFIER_CLASSES[name]

    get_class = mock.MagicMock(side_effect=resolve)
    monkeypatch.setattr(f"{DL_MODULE}.ReflectionHandler.get_class_by_name", get_class)
    return get_class


# --------------------------------------------------------------------------
# data fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def counts():
    rng = np.random.default_rng(0)
    return rng.poisson(1.5, size=(N_OBS, 5)).astype(np.float32)


@pytest.fixture
def sparse_counts(counts):
    return sp.csr_matrix(counts)


@pytest.fixture
def mdata_two(sparse_counts):
    rng = np.random.default_rng(1)
    rna = make_adata(sparse_counts.copy())
    atac = make_adata(
        sp.csr_matrix(rng.integers(0, 4, size=(N_OBS, 4)).astype(np.float32))
    )
    return make_mdata({"rna": rna, "atac": atac})
