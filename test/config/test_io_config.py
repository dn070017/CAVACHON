import os

import pytest

from cavachon.config.io_config import IOConfig


def test_io_config_defaults():
    config = IOConfig()
    expected_path = os.path.realpath(os.path.dirname("./"))
    assert config.datadir == expected_path
    assert config.outdir == expected_path


def test_io_config_custom_paths(tmp_path):
    datadir = os.path.realpath(os.path.join(tmp_path, "test_datadir"))
    outdir = os.path.realpath(os.path.join(tmp_path, "test_outdir"))
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # test with relative paths pointing to existing dirs
    relative_data_path = os.path.realpath(datadir)
    relative_out_path = os.path.realpath(outdir)
    config = IOConfig(datadir=str(relative_data_path), outdir=str(relative_out_path))
    assert config.datadir == os.path.realpath(relative_data_path)
    assert config.outdir == os.path.realpath(relative_out_path)

    # Test with paths pointing to non-existing dirs (validator should still resolve the parent)
    non_existent_datadir = os.path.realpath(
        os.path.join(tmp_path, "test_non_existent_datadir")
    )
    non_existent_outdir = os.path.realpath(
        os.path.join(tmp_path, "test_non_existent_outdir")
    )
    with pytest.raises(ValueError):
        IOConfig(datadir=non_existent_datadir)

    with pytest.raises(ValueError):
        IOConfig(outdir=non_existent_outdir)


def test_io_config_assignment(tmp_path):
    config = IOConfig()
    expected_default_path = os.path.realpath(os.path.dirname("./"))
    assert config.datadir == expected_default_path

    assigned_datadir = os.path.realpath(os.path.join(tmp_path, "test_assigned_datadir"))
    os.makedirs(assigned_datadir, exist_ok=True)
    config.datadir = assigned_datadir
    print(config.datadir, assigned_datadir)
    assert config.datadir == assigned_datadir
