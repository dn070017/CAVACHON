"""Shared fakes for the ``InteractiveVisualization`` and
``EnrichmentAnalysis`` unit tests.

Package boundary: the two modules under test. Every name they import from
another cavachon module is replaced where it is looked up:

* ``cavachon.tools.interactive_visualization.{AttributionAnalysis,
  ClusterAnalysis, DifferentialAnalysis}`` -> ``MagicMock``
* ``cavachon.tools.enrichment_analysis.InteractiveVisualization`` ->
  ``MagicMock``
* ``cavachon.tools.enrichment_analysis.DataFrameUtils`` -> ``MagicMock``
  whose ``filter_by_threshold`` compares like the real one, so tests can
  both see the call and get a usable mask.
* ``cavachon.tools.enrichment_analysis.gseapy`` -> ``MagicMock`` (network
  and output directories are never touched).

Heavy third-party reducers (``TSNE``, ``umap.UMAP``) are replaced by a
deterministic ``FakeReducer``; ``PCA`` stays real.

Side effects are blocked for every test: ``Figure.show``,
``Figure.write_html`` / ``write_image``, ``Path.mkdir``,
``tempfile.TemporaryDirectory``, ``shutil.copytree`` and ``pd.read_csv``.
"""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

IV_MODULE = "cavachon.tools.interactive_visualization"
EA_MODULE = "cavachon.tools.enrichment_analysis"
STAGING_DIR = "/staging/enrichment-analysis-tmp"


class Recorder:
    """Record (self, args, kwargs) for methods patched on a class."""

    def __init__(self):
        self.calls = []

    def __call__(self, owner, *args, **kwargs):
        self.calls.append((owner, args, kwargs))

    @property
    def args(self):
        return [c[1] for c in self.calls]


def _method(recorder):
    def method(self, *args, **kwargs):
        return recorder(self, *args, **kwargs)

    return method


class FakeReducer:
    """Deterministic stand-in for TSNE / UMAP: first two columns * 10."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_inputs = []
        FakeReducer.instances.append(self)

    def fit_transform(self, matrix):
        self.fit_inputs.append(np.asarray(matrix))
        return np.asarray(matrix)[:, :2] * 10.0


@pytest.fixture(autouse=True)
def figure_io(monkeypatch):
    """Block Figure.show / write_html / write_image and record calls."""
    recorders = SimpleNamespace(
        show=Recorder(), write_html=Recorder(), write_image=Recorder()
    )
    for name in ("show", "write_html", "write_image"):
        monkeypatch.setattr(go.Figure, name, _method(getattr(recorders, name)))
    return recorders


@pytest.fixture(autouse=True)
def iv_deps(monkeypatch):
    mocks = {}
    for name in ("AttributionAnalysis", "ClusterAnalysis", "DifferentialAnalysis"):
        m = mock.MagicMock(name=name)
        monkeypatch.setattr(f"{IV_MODULE}.{name}", m)
        mocks[name] = m
    return SimpleNamespace(**mocks)


@pytest.fixture(autouse=True)
def reducers(monkeypatch):
    FakeReducer.instances = []
    tsne = mock.MagicMock(name="TSNE", side_effect=FakeReducer)
    umap_cls = mock.MagicMock(name="UMAP", side_effect=FakeReducer)
    monkeypatch.setattr(f"{IV_MODULE}.TSNE", tsne)
    monkeypatch.setattr(f"{IV_MODULE}.umap.UMAP", umap_cls)
    return SimpleNamespace(TSNE=tsne, UMAP=umap_cls, instances=FakeReducer.instances)


def _fake_filter(data, threshold, operator, use_absolute=False):
    if use_absolute:
        data = data.abs()
    return {
        "leq": data <= threshold,
        "geq": data >= threshold,
        "lt": data < threshold,
        "gt": data > threshold,
    }[operator]


@pytest.fixture(autouse=True)
def ea_deps(monkeypatch):
    iv = mock.MagicMock(name="InteractiveVisualization")
    iv.prerank_ringplot.return_value = "RINGPLOT"
    iv.prerank_enrichment_score.side_effect = lambda result, term, filename: (
        f"ES:{term}"
    )
    dfu = mock.MagicMock(name="DataFrameUtils")
    dfu.filter_by_threshold.side_effect = _fake_filter
    gseapy = mock.MagicMock(name="gseapy")
    monkeypatch.setattr(f"{EA_MODULE}.InteractiveVisualization", iv)
    monkeypatch.setattr(f"{EA_MODULE}.DataFrameUtils", dfu)
    monkeypatch.setattr(f"{EA_MODULE}.gseapy", gseapy)
    return SimpleNamespace(
        InteractiveVisualization=iv, DataFrameUtils=dfu, gseapy=gseapy
    )


class FakeTemporaryDirectory:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        FakeTemporaryDirectory.instances.append(self)

    def __enter__(self):
        return STAGING_DIR

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def fs(monkeypatch):
    """Block file-system side effects used by EnrichmentAnalysis."""
    mkdir = Recorder()
    monkeypatch.setattr(Path, "mkdir", _method(mkdir))
    FakeTemporaryDirectory.instances = []
    monkeypatch.setattr(tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    copytree = mock.MagicMock(name="copytree")
    monkeypatch.setattr(shutil, "copytree", copytree)
    read_csv = mock.MagicMock(name="read_csv")
    monkeypatch.setattr(pd, "read_csv", read_csv)
    return SimpleNamespace(
        mkdir=mkdir,
        copytree=copytree,
        read_csv=read_csv,
        tempdirs=FakeTemporaryDirectory.instances,
    )


def make_prerank_result(res2d=None, results=None, ranking=None):
    """Plain stand-in for gseapy's Prerank result object."""
    if res2d is None:
        res2d = pd.DataFrame(
            {
                "Term": ["T1", "T2", "T3"],
                "NES": [2.0, -1.5, 0.5],
                "FDR q-val": [0.01, 0.04, 0.5],
                "Gene %": ["50.00%", "25.00%", "100.00%"],
            }
        )
    if results is None:
        results = {term: {"nes": 1.0} for term in res2d.get("Term", [])}
    return SimpleNamespace(res2d=res2d, results=results, ranking=ranking)
