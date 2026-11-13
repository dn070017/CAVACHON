from unittest.mock import call

import pytest

from cavachon.app import cavachon_application as app_mod
from cavachon.app.cavachon_application import CavachonApplication


def parse(set_argv, *argv):
    set_argv(*argv)
    app = CavachonApplication()
    app.parse_arguments()
    return app


def test_initial_args_none():
    assert CavachonApplication().args is None


def test_single_config_resolved_via_realpath(set_argv, mock_realpath):
    app = parse(set_argv, "config.yaml")
    mock_realpath.assert_called_once_with("config.yaml")
    assert app.args.configs == ["/resolved/config.yaml"]


def test_multiple_configs_keep_order(set_argv, mock_realpath):
    app = parse(set_argv, "b.yaml", "a.yaml", "/abs/c.yaml")
    assert mock_realpath.call_args_list == [
        call("b.yaml"),
        call("a.yaml"),
        call("/abs/c.yaml"),
    ]
    assert app.args.configs == [
        "/resolved/b.yaml",
        "/resolved/a.yaml",
        "/resolved//abs/c.yaml",
    ]


def test_missing_config_errors(set_argv, mock_realpath, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(set_argv)
    assert exc.value.code == 2
    assert "config.yaml" in capsys.readouterr().err
    mock_realpath.assert_not_called()


def test_help_exits_zero(set_argv, mock_realpath, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(set_argv, "--help")
    assert exc.value.code == 0
    assert "config files" in capsys.readouterr().out


def test_unknown_option_errors(set_argv, mock_realpath):
    with pytest.raises(SystemExit):
        parse(set_argv, "a.yaml", "--bogus")


def test_run_creates_and_runs_workflow_per_config(
    set_argv, mock_realpath, fake_workflow
):
    app = parse(set_argv, "a.yaml", "b.yaml")
    app.run()
    assert fake_workflow.call_args_list == [
        call("/resolved/a.yaml"),
        call("/resolved/b.yaml"),
    ]
    assert fake_workflow.return_value.run.call_count == 2


def test_run_stops_on_first_failure(set_argv, mock_realpath, fake_workflow):
    fake_workflow.return_value.run.side_effect = RuntimeError("boom")
    app = parse(set_argv, "a.yaml", "b.yaml")
    with pytest.raises(RuntimeError, match="boom"):
        app.run()
    fake_workflow.assert_called_once_with("/resolved/a.yaml")


def test_run_without_parse_raises(fake_workflow):
    with pytest.raises(AttributeError):
        CavachonApplication().run()
    fake_workflow.assert_not_called()


def test_main(set_argv, mock_realpath, fake_workflow):
    set_argv("x.yaml")
    app_mod.main()
    fake_workflow.assert_called_once_with("/resolved/x.yaml")
    fake_workflow.return_value.run.assert_called_once_with()


def test_config_existence_not_checked_at_parse_time(set_argv, mock_os):
    # Current behaviour: the CLI never checks existence; errors are deferred
    # to Workflow.
    app = parse(set_argv, "missing.yaml")
    assert app.args.configs == ["/resolved/missing.yaml"]
    mock_os.path.exists.assert_not_called()
    mock_os.path.isfile.assert_not_called()
