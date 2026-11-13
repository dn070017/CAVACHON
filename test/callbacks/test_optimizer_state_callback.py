"""Unit tests for cavachon.callbacks.optimizer_state_callback.

Mocked: the model (SimpleNamespace) and its optimizer (MagicMock with
get_weights / set_weights). The two xfail tests use a real Keras 3
optimizer / a tiny real tf.keras model, because the bugs are about the
callback's assumptions on the Keras 3 API.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.callbacks.optimizer_state_callback import OptimizerStateCallback


def _fake_optimizer():
    counter = {"n": 0}
    opt = MagicMock(name="optimizer")

    def get_weights():
        counter["n"] += 1
        return [np.array(counter["n"])]

    opt.get_weights.side_effect = get_weights
    return opt


def _early_stopping(best_epoch):
    es = tf.keras.callbacks.EarlyStopping(monitor="loss")
    es.best_epoch = best_epoch
    return es


def test_snapshot_taken_each_epoch():
    opt = _fake_optimizer()
    cb = OptimizerStateCallback()
    cb.set_model(SimpleNamespace(optimizer=opt))
    for epoch in range(3):
        cb.on_epoch_end(epoch)
    assert sorted(cb._opt_states) == [0, 1, 2]
    assert [int(v[0]) for v in cb._opt_states.values()] == [1, 2, 3]


@pytest.mark.parametrize(
    "model", [SimpleNamespace(), SimpleNamespace(optimizer=None)]
)
def test_no_snapshot_without_optimizer(model):
    cb = OptimizerStateCallback()
    cb.set_model(model)
    cb.on_epoch_end(0)
    cb.on_train_end()
    assert cb._opt_states == {}


def test_get_weights_error_swallowed():
    opt = MagicMock()
    opt.get_weights.side_effect = RuntimeError("boom")
    cb = OptimizerStateCallback()
    cb.set_model(SimpleNamespace(optimizer=opt))
    cb.on_epoch_end(0)
    assert cb._opt_states == {}


def test_restores_snapshot_of_best_epoch_from_first_early_stopping():
    opt = _fake_optimizer()
    other = MagicMock(spec=tf.keras.callbacks.Callback)
    model = SimpleNamespace(
        optimizer=opt,
        _callbacks=[other, _early_stopping(1), _early_stopping(2)],
    )
    cb = OptimizerStateCallback()
    cb.set_model(model)
    for epoch in range(3):
        cb.on_epoch_end(epoch)
    cb.on_train_end()
    opt.set_weights.assert_called_once()
    (restored,), _ = opt.set_weights.call_args
    assert restored is cb._opt_states[1]
    assert int(restored[0]) == 2


def test_no_restore_when_best_epoch_not_snapshotted():
    opt = _fake_optimizer()
    model = SimpleNamespace(optimizer=opt, _callbacks=[_early_stopping(5)])
    cb = OptimizerStateCallback()
    cb.set_model(model)
    cb.on_epoch_end(0)
    cb.on_train_end()
    opt.set_weights.assert_not_called()


def test_no_restore_without_early_stopping():
    opt = _fake_optimizer()
    model = SimpleNamespace(optimizer=opt, _callbacks=None)
    cb = OptimizerStateCallback()
    cb.set_model(model)
    cb.on_epoch_end(0)
    cb.on_train_end()
    opt.set_weights.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: Keras 3 optimizers have no get_weights(); the AttributeError "
    "is swallowed so no optimizer state is ever snapshotted",
)
def test_snapshot_works_with_keras3_optimizer():
    var = tf.Variable([1.0, 2.0])
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt.apply_gradients([(tf.constant([0.5, 0.5]), var)])
    cb = OptimizerStateCallback()
    cb.set_model(SimpleNamespace(optimizer=opt))
    cb.on_epoch_end(0)
    assert 0 in cb._opt_states


@pytest.mark.xfail(
    strict=True,
    reason="BUG: looks up EarlyStopping via model._callbacks, which Keras 3 "
    "models never have, so the optimizer state is never restored",
)
def test_restores_best_epoch_state_in_real_fit():
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse")
    # Isolate the second bug: give the optimizer a working get/set_weights.
    opt = model.optimizer
    snapshots = []

    def get_weights():
        snapshots.append([v.numpy().copy() for v in opt.variables])
        return snapshots[-1]

    opt.get_weights = get_weights
    opt.set_weights = MagicMock()
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=1, min_delta=1e9, restore_best_weights=True
    )
    cb = OptimizerStateCallback()
    x = np.ones((4, 2), dtype="float32")
    model.fit(x, x[:, :1], epochs=5, verbose=0, callbacks=[es, cb])
    assert es.stopped_epoch == 1 and es.best_epoch == 0
    assert set(cb._opt_states) == {0, 1}
    opt.set_weights.assert_called_once()
    (restored,), _ = opt.set_weights.call_args
    assert restored is cb._opt_states[0]
