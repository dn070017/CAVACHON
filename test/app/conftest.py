import sys
from unittest.mock import MagicMock

import pytest

from cavachon.app import cavachon_application as app_mod


@pytest.fixture
def fake_workflow(monkeypatch):
    """Mock cavachon.workflow.Workflow where the CLI looks it up."""
    mock = MagicMock(name="Workflow")
    monkeypatch.setattr(app_mod, "Workflow", mock)
    return mock


@pytest.fixture
def mock_os(monkeypatch):
    """Replace the `os` module seen by the CLI module only (so argparse /
    gettext keep the real one). No filesystem access: realpath is a pure
    stub and exists() would report False if anything consulted it."""
    fake_os = MagicMock(name="os")
    fake_os.path.realpath.side_effect = lambda p: f"/resolved/{p}"
    fake_os.path.exists.return_value = False
    monkeypatch.setattr(app_mod, "os", fake_os)
    return fake_os


@pytest.fixture
def mock_realpath(mock_os):
    return mock_os.path.realpath


@pytest.fixture
def set_argv(monkeypatch):
    def _set(*argv):
        monkeypatch.setattr(sys, "argv", ["cavachon", *argv])

    return _set
