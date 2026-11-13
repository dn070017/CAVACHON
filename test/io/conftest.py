"""Shared mocks for ``cavachon.io`` unit tests.

- ``pd.read_csv`` is replaced with a fake that parses in-memory text
  (registered per path) with the *real* pandas parser, so header /
  index_col / sep semantics are honoured while no file is opened.
- ``mmread`` is replaced with a fake returning in-memory matrices.
- ``os`` in file_reader is replaced so ``realpath`` is a recording identity.
- Config objects are SimpleNamespace stand-ins (no cavachon.config models).
"""

import io
import os
import posixpath
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import scipy.sparse as sp

import cavachon.io.file_reader as file_reader_module

_REAL_READ_CSV = pd.read_csv


class FakeFS:
    """In-memory registry of table text and matrices keyed by path."""

    def __init__(self):
        self.tables = {}
        self.matrices = {}
        self.read_csv = MagicMock(name="read_csv", side_effect=self._read_csv)
        self.mmread = MagicMock(name="mmread", side_effect=self._mmread)
        self.realpath = MagicMock(name="realpath", side_effect=posixpath.normpath)

    def _read_csv(self, filename, **kwargs):
        if filename not in self.tables:
            raise FileNotFoundError(filename)
        return _REAL_READ_CSV(io.StringIO(self.tables[filename]), **kwargs)

    def _mmread(self, filename):
        if filename not in self.matrices:
            raise FileNotFoundError(filename)
        m = self.matrices[filename]
        return sp.coo_matrix(m) if sp.issparse(m) else m

    def add_sample(self, directory, name, matrix, barcodes, features, transpose=False):
        m = matrix.T if transpose else matrix
        self.matrices[f"{directory}/{name}.mtx"] = sp.coo_matrix(m)
        self.tables[f"{directory}/{name}_b.tsv"] = "".join(f"{b}\n" for b in barcodes)
        self.tables[f"{directory}/{name}_f.tsv"] = "".join(
            f"{f}\t{f.upper()}\n" for f in features
        )


@pytest.fixture
def fake_fs(monkeypatch):
    fs = FakeFS()
    monkeypatch.setattr(file_reader_module.pd, "read_csv", fs.read_csv)
    monkeypatch.setattr(file_reader_module, "mmread", fs.mmread)
    monkeypatch.setattr(
        file_reader_module,
        "os",
        SimpleNamespace(path=SimpleNamespace(realpath=fs.realpath, join=os.path.join)),
    )
    return fs


def make_feature(filename, colnames=None, has_headers=False):
    # ``colnames=[]`` mirrors the ModalityFileFeatureConfig default.
    return SimpleNamespace(
        filename=filename,
        has_headers=has_headers,
        colnames=[] if colnames is None else colnames,
    )


def make_sample_modality(name, modality="RNA", transpose=False, colnames=True, prefix=""):
    return SimpleNamespace(
        name=modality,
        matrix=SimpleNamespace(filename=f"{prefix}{name}.mtx", transpose=transpose),
        barcodes=make_feature(f"{prefix}{name}_b.tsv", ["barcode"] if colnames else None),
        features=make_feature(f"{prefix}{name}_f.tsv", ["id", "symbol"] if colnames else None),
    )


def make_config(samples, modality_samples, datadir="/data"):
    """Stand-in for ApplicationConfig with only what FileReader reads.

    samples: {sample_name: (description, [sample_modality, ...])}
    modality_samples: {modality_name: [sample_name, ...]}
    """
    return SimpleNamespace(
        io=SimpleNamespace(datadir=datadir),
        modality={m: SimpleNamespace(samples=list(s)) for m, s in modality_samples.items()},
        sample={
            n: SimpleNamespace(description=d, modalities=mods)
            for n, (d, mods) in samples.items()
        },
    )


@pytest.fixture
def helpers():
    return SimpleNamespace(
        make_config=make_config,
        make_sample_modality=make_sample_modality,
        make_feature=make_feature,
    )
