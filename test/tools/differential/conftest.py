"""Shared fakes for the differential-analysis tools unit tests.

Package boundary: ``cavachon.tools``. Under test:

* ``cavachon.tools.differential_analysis``             (DifferentialAnalysis)
* ``cavachon.tools.hierarchical_differential_analysis`` (HierarchicalDifferentialAnalysis)
* ``cavachon.tools.differential_attribution_analysis``  (DifferentialAttributionAnalysis)

Every name these modules import from another cavachon package is replaced
where it is looked up:

* ``DataLoader``         -> ``FakeDataLoader`` (records its constructor
                            arguments, builds a real tf.data.Dataset from
                            the dense ``X`` and an optional numeric
                            ``obs["batch"]`` column of each modality)
* ``ReflectionHandler``  -> ``MagicMock`` whose ``get_class_by_name``
                            returns ``FakeDistribution`` (mean = parameters)

The model is a ``FakeModel`` (passed in), deterministic and differentiable:

    z[c]      = X_RNA @ E[c]                       (encode)
    z_hat     = z                                  (hierarchical_encode)
    x_params  = link(sum_c z_hat[c] @ W[c]) + batch_effect_RNA   (decode)

``cavachon.environment.Constants`` stays real. No file I/O.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata
import muon as mu
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants

DA_MODULE = "cavachon.tools.differential_analysis"
HDA_MODULE = "cavachon.tools.hierarchical_differential_analysis"
DAA_MODULE = "cavachon.tools.differential_attribution_analysis"

X_KEY = Constants.TENSOR_NAME_X
BATCH_KEY = Constants.TENSOR_NAME_BATCH


# ---------------------------------------------------------------------------
# DataLoader / ReflectionHandler fakes
# ---------------------------------------------------------------------------


class FakeDataLoader:
    """Stand-in for cavachon.dataloader.dataloader.DataLoader."""

    instances: list = []

    def __init__(
        self,
        mdata,
        batch_size=128,
        batch_effect_colnames=None,
        distribution_names=None,
        batch_effect_encoders=None,
    ):
        self.mdata = mdata
        self.batch_size = batch_size
        self.batch_effect_colnames = batch_effect_colnames
        self.distribution_names = distribution_names
        self.batch_effect_encoders = batch_effect_encoders
        tensors = {}
        for name, adata in mdata.mod.items():
            x = np.asarray(adata.X, dtype=np.float32).reshape(adata.n_obs, adata.n_vars)
            if "batch" in adata.obs.columns:
                be = adata.obs["batch"].to_numpy(dtype=np.float32).reshape(-1, 1)
            else:
                be = np.zeros((adata.n_obs, 1), dtype=np.float32)
            tensors[f"{name}_{X_KEY}"] = x
            tensors[f"{name}_{BATCH_KEY}"] = be
        self.dataset = tf.data.Dataset.from_tensor_slices(tensors)
        FakeDataLoader.instances.append(self)

    def __iter__(self):
        for batch in self.dataset.batch(self.batch_size):
            yield batch


class FakeDistribution:
    """Distribution whose mean is its parameter tensor."""

    def __init__(self, params):
        self.params = params

    @classmethod
    def from_parameterizer_output(cls, params):
        return cls(params)

    def mean(self):
        return self.params


@pytest.fixture
def fake_dataloader(monkeypatch):
    FakeDataLoader.instances = []
    for module in (DA_MODULE, HDA_MODULE, DAA_MODULE):
        monkeypatch.setattr(f"{module}.DataLoader", FakeDataLoader)
    return FakeDataLoader


@pytest.fixture
def reflection(monkeypatch):
    m = MagicMock(name="ReflectionHandler")
    m.get_class_by_name.return_value = FakeDistribution
    for module in (DA_MODULE, HDA_MODULE, DAA_MODULE):
        monkeypatch.setattr(f"{module}.ReflectionHandler", m)
    return m


@pytest.fixture(autouse=True)
def mocks(fake_dataloader, reflection):
    return SimpleNamespace(DataLoader=fake_dataloader, ReflectionHandler=reflection)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)
    tf.random.set_seed(0)


# ---------------------------------------------------------------------------
# Model fake
# ---------------------------------------------------------------------------


class FakeModel:
    """Deterministic, differentiable stand-in for cavachon's Model."""

    def __init__(self, E, W, modality="RNA", link=None, dist_name="FakeDist"):
        self.E = {k: tf.constant(v, dtype=tf.float32) for k, v in E.items()}
        self.W = {k: tf.constant(v, dtype=tf.float32) for k, v in W.items()}
        self.modality = modality
        self.link = link if link is not None else (lambda t: t)
        self.components = {
            name: SimpleNamespace(distribution_names={modality: dist_name})
            for name in E
        }
        self.encode_calls = []
        self.hier_calls = []
        self.decode_calls = []

    def encode(self, batch, training=False):
        self.encode_calls.append(training)
        x = tf.cast(batch[f"{self.modality}_{X_KEY}"], tf.float32)
        z = {name: tf.matmul(x, e) for name, e in self.E.items()}
        return {Constants.MODEL_OUTPUTS_Z: z}

    def hierarchical_encode(self, batch, z, training=False):
        self.hier_calls.append({k: np.array(v) for k, v in z.items()})
        return {Constants.MODEL_OUTPUTS_Z_HAT: dict(z)}

    def decode(self, batch, z_hat, components=None, training=False):
        self.decode_calls.append((list(components or []), training))
        total = tf.add_n([tf.matmul(z_hat[name], w) for name, w in self.W.items()])
        params = self.link(total) + tf.cast(
            batch[f"{self.modality}_{BATCH_KEY}"], tf.float32
        )
        key = f"{components[0]}_{self.modality}_{Constants.MODEL_OUTPUTS_X_PARAMS}"
        return {Constants.MODEL_OUTPUTS_X_PARAMS: {key: params}}


def identity_model(n_vars, components=("comp",)):
    """z[c] = X, x_params = X (only the first component decodes)."""
    eye = np.eye(n_vars, dtype=np.float32)
    E = {c: eye for c in components}
    W = {c: (eye if i == 0 else np.zeros_like(eye)) for i, c in enumerate(components)}
    return FakeModel(E, W)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_mdata(X, clusters=None, batch=None, atac=None, cluster_col="cluster"):
    """Tiny MuData. RNA has ``X``; optional ATAC has ``atac`` values."""
    X = np.asarray(X, dtype=np.float32)
    n_obs = X.shape[0]
    obs_names = [f"cell{i}" for i in range(n_obs)]
    obs = pd.DataFrame(index=obs_names)
    if clusters is not None:
        obs[cluster_col] = (
            clusters if isinstance(clusters, pd.Categorical) else list(clusters)
        )
    if batch is not None:
        obs["batch"] = np.asarray(batch, dtype=np.float32)
    mods = {
        "RNA": anndata.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]),
        )
    }
    if atac is not None:
        atac = np.asarray(atac, dtype=np.float32)
        mods["ATAC"] = anndata.AnnData(
            X=atac,
            obs=obs.copy(),
            var=pd.DataFrame(index=[f"p{j}" for j in range(atac.shape[1])]),
        )
    return mu.MuData(mods)
