"""Unit tests for cavachon.callbacks.early_stopping_callback.

Mocked: the Keras model (SimpleNamespace exposing get_weights /
set_weights / stop_training). The tf.keras.callbacks.EarlyStopping base
class is third-party and stays real.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock


from cavachon.callbacks.early_stopping_callback import EarlyStoppingCallback

RED = "\033[91m"
RESET = "\033[0m"


def _fake_model():
    counter = {"n": 0}

    def get_weights():
        counter["n"] += 1
        return [f"w{counter['n']}"]

    return SimpleNamespace(
        get_weights=get_weights, set_weights=MagicMock(), stop_training=False
    )


def _run(cb, losses):
    cb.on_train_begin()
    for epoch, loss in enumerate(losses):
        cb.on_epoch_end(epoch, {"loss": loss})
        if cb.model.stop_training:
            break
    cb.on_train_end()


def test_kwargs_forwarded_to_keras_early_stopping():
    cb = EarlyStoppingCallback(
        cumulative_offset=12,
        monitor="loss",
        min_delta=5,
        patience=3,
        restore_best_weights=True,
        verbose=0,
    )
    assert cb._cumulative_offset == 12
    assert cb.monitor == "loss"
    assert cb.patience == 3
    assert cb.min_delta == 5
    assert cb.restore_best_weights is True


def test_stops_on_plateau_and_restores_best_weights(capsys):
    cb = EarlyStoppingCallback(
        cumulative_offset=10, monitor="loss", patience=2, restore_best_weights=True
    )
    model = _fake_model()
    cb.set_model(model)
    # Lower is better for "loss": best at epoch 1, stop at epoch 3.
    _run(cb, [5.0, 1.0, 2.0, 3.0, 4.0])
    assert cb.best_epoch == 1
    assert cb.stopped_epoch == 3
    assert model.stop_training is True
    # epoch0 takes two snapshots (initial + improvement): w1, w2; epoch1 -> w3
    model.set_weights.assert_called_once_with(["w3"])
    out = capsys.readouterr().out
    assert out.startswith(RED)
    assert out.rstrip("\n").endswith(RESET)
    assert "Restoring Model Weights from Epoch" in out


def test_no_message_when_training_not_stopped(capsys):
    cb = EarlyStoppingCallback(monitor="loss", patience=5, restore_best_weights=True)
    model = _fake_model()
    cb.set_model(model)
    _run(cb, [3.0, 2.0, 1.0])
    assert cb.stopped_epoch == 0
    # Keras still restores best weights at train end ...
    model.set_weights.assert_called_once()
    # ... but the custom message is only for early stops.
    assert capsys.readouterr().out == ""


def test_no_message_without_restore_best_weights(capsys):
    cb = EarlyStoppingCallback(monitor="loss", patience=1, restore_best_weights=False)
    model = _fake_model()
    cb.set_model(model)
    _run(cb, [1.0, 2.0, 3.0])
    assert cb.stopped_epoch > 0
    model.set_weights.assert_not_called()
    assert capsys.readouterr().out == ""


def test_state_reset_between_fits(capsys):
    cb = EarlyStoppingCallback(monitor="loss", patience=1, restore_best_weights=True)
    cb.set_model(_fake_model())
    _run(cb, [1.0, 2.0, 3.0])
    assert cb.stopped_epoch == 1
    capsys.readouterr()
    cb.set_model(_fake_model())
    # Keras' on_train_begin resets wait/stopped_epoch/best_epoch but NOT
    # `best` (third-party behaviour), so the second fit must beat 1.0.
    _run(cb, [0.5, 0.4])  # improving -> never stops
    assert cb.stopped_epoch == 0
    assert cb.best_epoch == 1
    assert capsys.readouterr().out == ""


def test_restored_epoch_number_is_one_based(capsys):
    # regression: the epoch was printed 0-based (offset + best_epoch), one
    # lower than the "Total Epoch" VerboseCallback shows
    cb = EarlyStoppingCallback(
        cumulative_offset=10, monitor="loss", patience=2, restore_best_weights=True
    )
    cb.set_model(_fake_model())
    # Best is phase epoch index 1, which VerboseCallback announced as
    # "Phase Epoch: 2 ... Total Epoch: 12" (offset + epoch + 1).
    _run(cb, [5.0, 1.0, 2.0, 3.0])
    out = capsys.readouterr().out
    assert "Restoring Model Weights from Epoch 12" in out


def test_positional_monitor_is_respected():
    # regression: cumulative_offset came before *args and swallowed the
    # positional `monitor` argument of EarlyStopping
    cb = EarlyStoppingCallback("loss")
    assert cb.monitor == "loss"
    assert cb._cumulative_offset == 0
