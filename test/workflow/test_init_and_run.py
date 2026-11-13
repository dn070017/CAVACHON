"""Unit tests for Workflow.__init__ and Workflow.run.

Package boundary: ``cavachon.workflow``. ApplicationConfig.from_yaml,
AnnDataFilterHandler.from_config and Model.make are MagicMocks (conftest
``deps``); os.makedirs is a recorder. In ``run`` tests every Workflow step
method is replaced on the instance by a child of one manager Mock so the
call order can be asserted; mdata is a MagicMock whose ``write`` is
recorded.
"""

from unittest import mock

import pytest

from cavachon.workflow.workflow import Workflow
from test.workflow.conftest import make_config, make_workflow

STEPS = (
    "setup_mdata",
    "setup_dataloader",
    "setup_train_scheduler",
    "load_model_weights",
    "train_model",
    "predict",
    "perform_clustering_analysis",
    "visualize_embedding",
    "visualize_conditional_attribution_scores",
    "perform_differential_analysis",
    "perform_hierarchical_differential_analysis",
    "perform_differential_attribution_analysis",
    "perform_enrichment_analysis",
)


# --------------------------------------------------------------------------
# __init__
# --------------------------------------------------------------------------


def test_init_reads_config_from_yaml_and_builds_filters(deps):
    cfg = make_config()
    deps.ApplicationConfig.from_yaml.return_value = cfg

    wf = Workflow("path/to/config.yaml")

    deps.ApplicationConfig.from_yaml.assert_called_once_with("path/to/config.yaml")
    deps.AnnDataFilterHandler.from_config.assert_called_once_with(cfg)
    assert wf.config is cfg
    assert wf.anndata_filters is deps.AnnDataFilterHandler.from_config.return_value


def test_init_sets_empty_state(deps):
    deps.ApplicationConfig.from_yaml.return_value = make_config()
    wf = Workflow("c.yaml")

    assert wf.mdata is None
    assert wf.dataloader is None
    assert wf.model is None
    assert wf.train_scheduler is None
    assert wf.train_history == []
    assert wf.outputs == {}
    assert wf.differential_analysis_results == {}


def test_init_state_is_not_shared_between_instances(deps):
    deps.ApplicationConfig.from_yaml.return_value = make_config()
    a = Workflow("a.yaml")
    b = Workflow("b.yaml")

    a.train_history.append("h")
    a.outputs["z"] = 1
    a.differential_analysis_results[("RNA",)] = {}

    assert b.train_history == []
    assert b.outputs == {}
    assert b.differential_analysis_results == {}


def test_init_does_not_touch_data_or_model(deps):
    deps.ApplicationConfig.from_yaml.return_value = make_config()
    Workflow("c.yaml")

    deps.Model.make.assert_not_called()
    deps.MultiModality.assert_not_called()
    deps.DataLoader.assert_not_called()
    deps.FileReader.read_multiomics_data.assert_not_called()


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def wire_run(deps, makedirs, load_weights, train):
    config = make_config()
    config.model.load_weights = load_weights
    config.training.train = train
    config.components = ["component-configs"]
    config.model.name = "my_model"
    wf = make_workflow(config, model=None)

    manager = mock.Mock()
    for step in STEPS:
        setattr(wf, step, getattr(manager, step))
    manager.attach_mock(deps.Model.make, "make")
    manager.attach_mock(makedirs, "makedirs")
    wf.mdata = mock.MagicMock(name="mdata")
    manager.attach_mock(wf.mdata.write, "write")
    return wf, manager


def call_names(manager):
    return [c[0] for c in manager.mock_calls]


HEAD = ["setup_mdata", "setup_dataloader", "make", "setup_train_scheduler"]
TAIL = [
    "perform_clustering_analysis",
    "visualize_embedding",
    "makedirs",
    "write",
    "visualize_conditional_attribution_scores",
    "perform_differential_analysis",
    "perform_hierarchical_differential_analysis",
    "perform_differential_attribution_analysis",
    "perform_enrichment_analysis",
]


@pytest.mark.parametrize(
    "load_weights, train, middle",
    [
        (False, True, ["train_model", "predict"]),
        (False, False, ["predict"]),
        (True, False, ["predict", "load_model_weights", "predict"]),
        (True, True, ["predict", "load_model_weights", "train_model", "predict"]),
    ],
)
def test_run_step_order(deps, makedirs, load_weights, train, middle):
    wf, manager = wire_run(deps, makedirs, load_weights, train)
    wf.run()
    assert call_names(manager) == HEAD + middle + TAIL


def test_run_builds_model_from_config_components(deps, makedirs):
    wf, _ = wire_run(deps, makedirs, False, True)
    wf.run()

    deps.Model.make.assert_called_once_with(
        component_configs=["component-configs"], name="my_model"
    )
    assert wf.model is deps.Model.make.return_value


def test_run_writes_h5mu_under_outdir(deps, makedirs):
    wf, _ = wire_run(deps, makedirs, False, True)
    wf.run()

    makedirs.assert_called_once_with("/out/mdata", exist_ok=True)
    wf.mdata.write.assert_called_once_with("/out/mdata/mdata.h5mu")


def test_run_returns_none(deps, makedirs):
    wf, _ = wire_run(deps, makedirs, False, False)
    assert wf.run() is None


def test_run_propagates_step_failure_and_stops(deps, makedirs):
    wf, manager = wire_run(deps, makedirs, False, True)
    manager.train_model.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        wf.run()

    names = call_names(manager)
    assert names[-1] == "train_model"
    assert "write" not in names
    assert "predict" not in names
