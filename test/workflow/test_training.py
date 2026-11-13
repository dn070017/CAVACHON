"""Unit tests for Workflow.train_model, load_model_weights and predict.

Package boundary: ``cavachon.workflow``. The model, the training
scheduler and the dataloader are MagicMocks/SimpleNamespaces; os.makedirs
is a recorder (conftest). One test feeds a real tiny tf.data.Dataset to
check the shuffle/batch pipeline that train_model builds.
"""

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from test.workflow.conftest import make_config, make_workflow


def fake_component():
    return mock.MagicMock(trainable=True)


def training_workflow(shuffle=False, save_weights=True):
    config = make_config()
    config.dataset.shuffle = shuffle
    config.dataset.batch_size = 32
    config.training.max_regular_training_epochs = 11
    config.model.save_weights = save_weights
    wf = make_workflow(config)
    wf.mdata.n_obs = 1234
    wf.model.name = "my_model"
    wf.model.components = {"a": fake_component(), "b": fake_component()}
    return wf


# --------------------------------------------------------------------------
# train_model
# --------------------------------------------------------------------------


def test_train_model_without_shuffle_batches_and_fits():
    wf = training_workflow(shuffle=False)
    dataset = wf.dataloader.dataset

    wf.train_model()

    dataset.shuffle.assert_not_called()
    dataset.batch.assert_called_once_with(32)
    wf.train_scheduler.fit.assert_called_once_with(
        dataset.batch.return_value, epochs=11
    )
    assert wf.train_history is wf.train_scheduler.fit.return_value


def test_train_model_with_shuffle_uses_n_obs_buffer():
    wf = training_workflow(shuffle=True)
    dataset = wf.dataloader.dataset

    wf.train_model()

    dataset.shuffle.assert_called_once_with(1234)
    dataset.batch.assert_not_called()
    dataset.shuffle.return_value.batch.assert_called_once_with(32)
    wf.train_scheduler.fit.assert_called_once_with(
        dataset.shuffle.return_value.batch.return_value, epochs=11
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_train_model_real_dataset_pipeline(shuffle):
    wf = training_workflow(shuffle=shuffle)
    wf.config.dataset.batch_size = 4
    wf.mdata.n_obs = 10
    wf.dataloader.dataset = tf.data.Dataset.range(10)
    seen = {}

    def fit(x, epochs):
        seen["batches"] = [b.numpy() for b in x]
        return ["history"]

    wf.train_scheduler.fit.side_effect = fit
    wf.train_model()

    sizes = [len(b) for b in seen["batches"]]
    assert sizes == [4, 4, 2]
    values = np.concatenate(seen["batches"])
    assert sorted(values.tolist()) == list(range(10))
    if not shuffle:
        assert values.tolist() == list(range(10))
    assert wf.train_history == ["history"]


def test_train_model_saves_weights_when_enabled(makedirs):
    wf = training_workflow(save_weights=True)
    manager = mock.Mock()
    manager.attach_mock(makedirs, "makedirs")
    manager.attach_mock(wf.model.save_weights, "save_weights")

    wf.train_model()

    assert manager.mock_calls == [
        mock.call.makedirs("/ckpt", exist_ok=True),
        mock.call.save_weights("/ckpt/my_model.weights.h5"),
    ]


def test_train_model_does_not_save_when_disabled(makedirs):
    wf = training_workflow(save_weights=False)
    wf.train_model()

    makedirs.assert_not_called()
    wf.model.save_weights.assert_not_called()


def test_train_model_saves_after_fit(makedirs):
    wf = training_workflow(save_weights=True)
    manager = mock.Mock()
    manager.attach_mock(wf.train_scheduler.fit, "fit")
    manager.attach_mock(wf.model.save_weights, "save_weights")

    wf.train_model()

    assert [c[0] for c in manager.mock_calls] == ["fit", "save_weights"]


def test_train_model_freezes_components_and_finishes_scalers():
    wf = training_workflow()
    wf.train_model()

    for component in wf.model.components.values():
        assert component.trainable is False
        component.set_progressive_scaler_iteration.assert_called_once_with(
            current_iteration=1.0, total_iterations=1.0
        )
    assert wf.model.compile.called


# --------------------------------------------------------------------------
# load_model_weights
# --------------------------------------------------------------------------


def test_load_model_weights_path():
    wf = make_workflow()
    wf.model.name = "my_model"
    wf.load_model_weights()
    wf.model.load_weights.assert_called_once_with("/ckpt/my_model.weights.h5")


@pytest.mark.parametrize("error", [OSError("nope"), FileNotFoundError("missing")])
def test_load_model_weights_warns_on_os_error(error):
    wf = make_workflow()
    wf.model.load_weights.side_effect = error

    with pytest.warns(RuntimeWarning) as record:
        assert wf.load_model_weights() is None

    assert len(record) == 1
    assert str(record[0].message) == "Cannot load the pretrained weights in /ckpt."


def test_load_model_weights_other_errors_propagate():
    wf = make_workflow()
    wf.model.load_weights.side_effect = ValueError("shape mismatch")
    with pytest.raises(ValueError, match="shape mismatch"):
        wf.load_model_weights()


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


def test_predict_freezes_compiles_and_stores_outputs():
    config = make_config()
    config.dataset.batch_size = 99
    wf = make_workflow(config)
    wf.model.trainable = True
    manager = mock.Mock()
    manager.attach_mock(wf.model.compile, "compile")
    manager.attach_mock(wf.model.predict, "predict")

    wf.predict()

    assert wf.model.trainable is False
    assert manager.mock_calls == [
        mock.call.compile(),
        mock.call.predict(wf.mdata, batch_size=99),
    ]
    assert wf.outputs is wf.model.predict.return_value
