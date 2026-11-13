"""Shared mocks for cavachon.utils unit tests.

ReflectionHandler talks to three collaborators outside the utils package:
``Settings.src_path`` (cavachon.environment), ``glob.glob`` (filesystem) and
``import_module`` (module loading). All three are replaced where
reflection_handler looks them up.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import cavachon.utils.reflection_handler as reflection_module

FAKE_SRC = Path("/fake/root/cavachon")


class FakeGlob:
    """Stand-in for the ``glob`` module: records patterns, returns canned hits.

    ``hits`` maps an exact pattern to the list glob should return; every
    other pattern returns [].
    """

    def __init__(self):
        self.hits = {}
        self.calls = []

    def glob(self, pattern):
        self.calls.append(pattern)
        return list(self.hits.get(pattern, []))


class FakeImporter:
    """Stand-in for ``importlib.import_module`` returning fake modules."""

    def __init__(self):
        self.modules = {}
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        if name not in self.modules:
            raise ModuleNotFoundError(name)
        return self.modules[name]


@pytest.fixture
def fake_settings(monkeypatch):
    settings = SimpleNamespace(src_path=FAKE_SRC)
    monkeypatch.setattr(reflection_module, "Settings", settings)
    return settings


@pytest.fixture
def fake_glob(monkeypatch):
    fake = FakeGlob()
    monkeypatch.setattr(reflection_module, "glob", fake)
    return fake


@pytest.fixture
def fake_import(monkeypatch):
    fake = FakeImporter()
    monkeypatch.setattr(reflection_module, "import_module", fake)
    return fake


@pytest.fixture
def reflection_env(fake_settings, fake_glob, fake_import):
    """Settings, glob and import_module all mocked for ReflectionHandler."""
    return SimpleNamespace(src=fake_settings.src_path, glob=fake_glob, importer=fake_import)
