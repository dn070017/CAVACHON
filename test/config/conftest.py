"""Shared mocks for ``cavachon.config`` unit tests.

Boundary rules:
- ``cavachon.utils.GeneralUtils`` (another cavachon package) is mocked in
  every config module that looks it up.
- File access (``open``), YAML parsing and ``os.path`` calls are mocked;
  no test touches the filesystem.
- ``cavachon.environment.Constants`` and third-party libraries stay real.
"""

import copy
import os
import posixpath
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open

import pytest

import cavachon.config.application_config as application_config_module
import cavachon.config.base as base_module
import cavachon.config.component_config as component_config_module
import cavachon.config.io_config as io_config_module

FAKE_CWD = "/cwd"


def fake_tensorflow_compatible_str(s):
    """Simple stand-in for GeneralUtils.tensorflow_compatible_str."""
    return s.replace(" ", "_").replace(":", "_")


def fake_realpath(p):
    """Deterministic stand-in for os.path.realpath (no filesystem access)."""
    if not p.startswith("/"):
        p = f"{FAKE_CWD}/{p}"
    return posixpath.normpath(p)


@pytest.fixture(autouse=True)
def general_utils(monkeypatch):
    """Mock GeneralUtils where each config module looks it up.

    ``order_components`` defaults to returning the components in the
    order they were passed (dict insertion order).
    """
    utils = MagicMock(name="GeneralUtils")
    utils.tensorflow_compatible_str.side_effect = fake_tensorflow_compatible_str
    utils.order_components.side_effect = lambda d: list(d.values())
    for module in (base_module, component_config_module, application_config_module):
        monkeypatch.setattr(module, "GeneralUtils", utils)
    return utils


@pytest.fixture(autouse=True)
def fake_os(monkeypatch):
    """Replace ``os`` in io_config and application_config with stand-ins."""
    realpath = MagicMock(name="realpath", side_effect=fake_realpath)
    isdir = MagicMock(name="isdir", return_value=False)
    monkeypatch.setattr(
        io_config_module,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(realpath=realpath, dirname=os.path.dirname)
        ),
    )
    monkeypatch.setattr(
        application_config_module,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(realpath=realpath, join=posixpath.join, isdir=isdir)
        ),
    )
    return SimpleNamespace(realpath=realpath, isdir=isdir)


@pytest.fixture
def yaml_source(monkeypatch):
    """Mock ``open`` and ``yaml`` in application_config.

    Set ``yaml_source.data`` (a dict) before calling ``from_yaml``.
    """
    source = SimpleNamespace(data=None)
    full_loader = object()
    source.open = mock_open(read_data="<mocked>")
    source.load = MagicMock(name="yaml.load", side_effect=lambda f, Loader: source.data)
    source.full_loader = full_loader
    monkeypatch.setattr(application_config_module, "open", source.open, raising=False)
    monkeypatch.setattr(
        application_config_module,
        "yaml",
        SimpleNamespace(load=source.load, FullLoader=full_loader),
    )
    return source


@pytest.fixture
def base_config():
    """A minimal valid application config as a plain dict."""
    return copy.deepcopy(
        {
            "io": {"datadir": "/data", "outdir": "/out", "checkpointdir": "/ckpt"},
            "modalities": [
                {"name": "RNA", "type": "rna"},
                {"name": "ATAC", "type": "atac"},
            ],
            "model": {
                "components": [
                    {"name": "ATAC", "modalities": [{"name": "ATAC"}]},
                    {
                        "name": "RNA",
                        "conditioned_on_z_hat": ["ATAC"],
                        "modalities": [{"name": "RNA"}],
                    },
                ],
            },
        }
    )


@pytest.fixture
def load_config(yaml_source):
    """Load a dict through ApplicationConfig.from_yaml with mocked I/O."""
    from cavachon.config import ApplicationConfig

    def _load(content, filename="/cfg/config.yaml"):
        yaml_source.data = content
        return ApplicationConfig.from_yaml(filename)

    return _load
