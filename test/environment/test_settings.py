"""Settings tests.

Settings computes its paths at import time with os.path.realpath, which
resolves symlinks against the real filesystem. Path computation from
``__file__`` is tested in-process by re-executing the module into a fresh
module object (not replacing sys.modules) with ``os.path.realpath`` and
``os.getcwd`` patched, so no filesystem lookups or subprocesses are used.
"""

import importlib.util
import os
from pathlib import Path

import pytest

import cavachon.environment as environment_pkg
import cavachon.environment.settings as settings_module
from cavachon.environment import Settings

SETTINGS_FILE = os.path.abspath(settings_module.__file__)
EXPECTED_ROOT = os.path.normpath(os.path.join(os.path.dirname(SETTINGS_FILE), "../../"))


@pytest.fixture
def fresh_settings(monkeypatch):
    """Load settings.py into a new module with realpath/getcwd mocked."""
    realpath_calls = []

    def fake_realpath(p, *args, **kwargs):
        realpath_calls.append(p)
        return os.path.normpath(p)

    def load():
        spec = importlib.util.spec_from_file_location(
            "_settings_under_test", SETTINGS_FILE
        )
        module = importlib.util.module_from_spec(spec)
        with monkeypatch.context() as m:
            m.setattr(os.path, "realpath", fake_realpath)
            spec.loader.exec_module(module)
        return module.Settings, realpath_calls

    return load


def test_settings_reexported_from_package():
    assert environment_pkg.Settings is settings_module.Settings is Settings


def test_paths_are_absolute_path_objects():
    for attr in ("root_path", "test_path", "src_path", "tmp_path"):
        value = getattr(Settings, attr)
        assert isinstance(value, Path), attr
        assert value.is_absolute(), attr


def test_paths_are_resolved_without_dotdot():
    for attr in ("root_path", "test_path", "src_path", "tmp_path"):
        assert ".." not in getattr(Settings, attr).parts


def test_root_is_two_levels_above_settings_file(fresh_settings):
    s, _ = fresh_settings()
    assert s.root_path == Path(EXPECTED_ROOT)


def test_child_paths(fresh_settings):
    s, _ = fresh_settings()
    assert s.src_path == s.root_path / "cavachon"
    assert s.test_path == s.root_path / "test"
    assert s.tmp_path == s.root_path / "tmp"


def test_src_path_contains_settings_module(fresh_settings):
    s, _ = fresh_settings()
    assert Path(SETTINGS_FILE) == s.src_path / "environment" / "settings.py"


def test_realpath_used_for_every_path(fresh_settings):
    _, calls = fresh_settings()
    assert len(calls) == 4


def test_independent_of_cwd_and_env(fresh_settings, monkeypatch):
    baseline, _ = fresh_settings()
    monkeypatch.setattr(os, "getcwd", lambda: "/somewhere/else")
    monkeypatch.setenv("CAVACHON_ROOT", "/env/override")
    monkeypatch.setenv("PWD", "/somewhere/else")
    moved, _ = fresh_settings()
    for attr in ("root_path", "test_path", "src_path", "tmp_path"):
        assert getattr(moved, attr) == getattr(baseline, attr)


def test_symlinks_resolved_via_realpath(monkeypatch):
    """If realpath resolves a symlinked checkout, Settings uses the target."""
    spec = importlib.util.spec_from_file_location("_settings_symlink", SETTINGS_FILE)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as m:
        m.setattr(
            os.path,
            "realpath",
            lambda p, *a, **k: os.path.normpath(p).replace(
                EXPECTED_ROOT, "/resolved/repo", 1
            ),
        )
        spec.loader.exec_module(module)
    assert module.Settings.root_path == Path("/resolved/repo")
    assert module.Settings.src_path == Path("/resolved/repo/cavachon")


def test_module_settings_untouched_by_reloads(fresh_settings):
    before = Settings.src_path
    fresh_settings()
    assert settings_module.Settings is Settings
    assert Settings.src_path == before
