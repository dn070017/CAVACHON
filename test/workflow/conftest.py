"""Shared fakes for the ``cavachon.workflow`` unit tests.

Package boundary: ``cavachon.workflow``. Every name ``workflow.py``
imports from another cavachon package is replaced with a ``MagicMock``
where ``workflow.py`` looks it up (``cavachon.workflow.workflow.<name>``).
``cavachon.environment.Constants`` stays real (plain constants).

File-system side effects are blocked for every test: ``os.makedirs``,
``anndata.read_h5ad`` and ``pandas.DataFrame.to_csv`` are replaced by
recorders, so no test touches the disk.
"""

from types import SimpleNamespace
from unittest import mock

import anndata
import muon as mu
import numpy as np
import pandas as pd
import pytest

from cavachon.workflow.workflow import Workflow

WORKFLOW_MODULE = "cavachon.workflow.workflow"

# every name workflow.py imports from another cavachon package
MOCKED_NAMES = (
    "ApplicationConfig",
    "DataLoader",
    "AnnDataFilterHandler",
    "FileReader",
    "Modality",
    "MultiModality",
    "Model",
    "SequentialTrainingScheduler",
    "ClusterAnalysis",
    "DifferentialAnalysis",
    "DifferentialAttributionAnalysis",
    "HierarchicalDifferentialAnalysis",
    "EnrichmentAnalysis",
    "_safe_filename",
    "InteractiveVisualization",
    "AnnDataUtils",
)


@pytest.fixture(autouse=True)
def deps(monkeypatch):
    """Replace every cavachon collaborator of workflow.py with a MagicMock.

    ``_safe_filename`` gets a recognisable side effect so tests can see
    that (and where) it was applied.
    """
    mocks = {}
    for name in MOCKED_NAMES:
        m = mock.MagicMock(name=name)
        monkeypatch.setattr(f"{WORKFLOW_MODULE}.{name}", m)
        mocks[name] = m
    mocks["_safe_filename"].side_effect = lambda v: f"Safe-{v}"
    return SimpleNamespace(**mocks)


@pytest.fixture(autouse=True)
def makedirs(monkeypatch):
    """Record os.makedirs calls instead of creating directories."""
    m = mock.MagicMock(name="makedirs")
    monkeypatch.setattr("os.makedirs", m)
    return m


@pytest.fixture(autouse=True)
def read_h5ad(monkeypatch):
    m = mock.MagicMock(name="read_h5ad")
    monkeypatch.setattr(anndata, "read_h5ad", m)
    return m


@pytest.fixture(autouse=True)
def to_csv_calls(monkeypatch):
    """Record DataFrame.to_csv as (frame copy, path, kwargs) tuples."""
    calls = []

    def fake_to_csv(self, path=None, *args, **kwargs):
        calls.append((self.copy(), path, kwargs))

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    return calls


def make_config(**sections):
    """Build a SimpleNamespace stand-in for ApplicationConfig.

    Only the attributes Workflow reads are provided. Each keyword
    replaces the whole section of the same name.
    """
    analysis = SimpleNamespace(
        clustering=[],
        differential_analysis=[],
        visualize_embedding=[],
        conditional_attribution_scores=[],
        differential_attribution_analysis=[],
        hierarchical_differential_analysis=[],
        enrichment_analysis=[],
    )
    config = SimpleNamespace(
        io=SimpleNamespace(outdir="/out", checkpointdir="/ckpt", datadir="/data"),
        dataset=SimpleNamespace(batch_size=64, shuffle=False),
        training=SimpleNamespace(
            train=True,
            max_regular_training_epochs=7,
            optimizer=SimpleNamespace(name="adam", learning_rate=1e-3),
            early_stopping=True,
        ),
        model=SimpleNamespace(
            name="cavachon",
            load_weights=False,
            save_weights=True,
            components=[],
        ),
        modality_names=[],
        modality={},
        components=[],
        analysis=analysis,
    )
    for key, value in sections.items():
        setattr(config, key, value)
    return config


def make_workflow(config=None, **attrs):
    """Build a Workflow without running __init__ (no ApplicationConfig)."""
    wf = object.__new__(Workflow)
    wf.config = config if config is not None else make_config()
    wf.mdata = mock.MagicMock(name="mdata")
    wf.dataloader = SimpleNamespace(
        batch_size=64,
        batch_effect_encoders={"RNA": {"batch": "encoder"}},
        n_vars_batch_effect={},
        dataset=mock.MagicMock(name="dataset"),
    )
    wf.model = mock.MagicMock(name="model")
    wf.model.name = "cavachon"
    wf.train_scheduler = mock.MagicMock(name="train_scheduler")
    wf.train_history = []
    wf.outputs = {}
    wf.differential_analysis_results = {}
    wf.batch_effect_colnames = {"RNA": ["batch"]}
    wf.distribution_names = {"RNA": "IndependentZeroInflatedNegativeBinomial"}
    wf.anndata_filters = mock.MagicMock(name="anndata_filters")
    for key, value in attrs.items():
        setattr(wf, key, value)
    return wf


def make_mdata(obs_columns=None, n_vars=None, n_obs=6):
    """Tiny in-memory MuData. ``obs_columns`` maps modality -> {col: values}."""
    n_vars = n_vars or {"RNA": 3}
    obs_columns = obs_columns or {}
    obs_names = [f"cell{i}" for i in range(n_obs)]
    mods = {}
    for name, nv in n_vars.items():
        obs = pd.DataFrame(obs_columns.get(name, {}), index=obs_names)
        var = pd.DataFrame(index=[f"{name}_v{j}" for j in range(nv)])
        mods[name] = anndata.AnnData(
            X=np.zeros((n_obs, nv), dtype=np.float32), obs=obs, var=var
        )
    return mu.MuData(mods)
