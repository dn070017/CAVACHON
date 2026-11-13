import pytest

from cavachon.config import IOConfig


def test_absolute_paths_normalized(fake_os):
    io = IOConfig(datadir="/x/a/../b/", outdir="/x", checkpointdir="/x/c")
    assert io.datadir == "/x/b"
    assert io.outdir == "/x"
    assert io.checkpointdir == "/x/c"


@pytest.mark.parametrize(
    "value, passed",
    [("/x/data", "/x/data"), ("/x/data/", "/x/data"), ("data", "data"), ("./", ".")],
)
def test_trailing_slash_stripped_before_realpath(fake_os, value, passed):
    IOConfig(datadir=value)
    fake_os.realpath.assert_called_once_with(passed)


def test_realpath_result_is_stored(fake_os):
    fake_os.realpath.side_effect = lambda p: "/resolved/target"
    assert IOConfig(datadir="/link").datadir == "/resolved/target"


def test_relative_paths_resolved_without_any_base_dir(fake_os):
    # The relative value is handed to realpath as-is, i.e. resolved
    # against the process cwd (here the fake "/cwd").
    io = IOConfig(datadir="data", outdir="./out/")
    assert io.datadir == "/cwd/data"
    assert io.outdir == "/cwd/out"


def test_tilde_not_expanded(fake_os):
    # Documents current behaviour: ``~`` is passed literally.
    IOConfig(datadir="~/x")
    fake_os.realpath.assert_called_once_with("~/x")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: defaults ('./') bypass the path validator and stay "
    "relative, unlike an explicit './' which is resolved",
)
def test_default_paths_resolved_like_explicit(fake_os):
    default = IOConfig()
    explicit = IOConfig(checkpointdir="./", datadir="./", outdir="./")
    assert default.model_dump() == explicit.model_dump()
