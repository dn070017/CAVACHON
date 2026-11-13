"""Unit tests for Workflow.setup_train_scheduler.

Package boundary: ``cavachon.workflow``. SequentialTrainingScheduler and
every object the workflow passes to it are replaced by fakes.
"""

from types import SimpleNamespace
from unittest import mock

from cavachon.workflow.workflow import Workflow


def make_workflow():
    wf = object.__new__(Workflow)
    wf.config = SimpleNamespace(
        training=SimpleNamespace(
            optimizer=SimpleNamespace(name="adam", learning_rate=5e-4),
            early_stopping=True,
        ),
        io=SimpleNamespace(outdir="/out"),
    )
    wf.model = object()
    wf.mdata = object()
    wf.dataloader = SimpleNamespace(batch_size=32)
    wf.distribution_names = {"RNA": "IndependentZeroInflatedNegativeBinomial"}
    wf.batch_effect_colnames = {"RNA": ["batch"]}
    return wf


def test_scheduler_receives_distribution_names_and_batch_effect_colnames(monkeypatch):
    # regression: distribution_names and batch_effect_colnames were passed
    # positionally in the opposite order of the scheduler's signature
    scheduler_cls = mock.MagicMock()
    monkeypatch.setattr(
        "cavachon.workflow.workflow.SequentialTrainingScheduler", scheduler_cls
    )
    wf = make_workflow()
    wf.setup_train_scheduler()

    args, kwargs = scheduler_cls.call_args
    assert args == (wf.model, wf.mdata, "adam", 5e-4, True, 32)
    assert kwargs == {
        "outdir": "/out",
        "distribution_names": wf.distribution_names,
        "batch_effect_colnames": wf.batch_effect_colnames,
    }
    assert wf.train_scheduler is scheduler_cls.return_value


def test_keyword_names_match_scheduler_signature():
    import inspect

    from cavachon.scheduler.sequential_training_scheduler import (
        SequentialTrainingScheduler,
    )

    params = inspect.signature(SequentialTrainingScheduler.__init__).parameters
    for name in ("outdir", "distribution_names", "batch_effect_colnames"):
        assert name in params
