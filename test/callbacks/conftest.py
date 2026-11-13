"""Shared fakes for cavachon.callbacks unit tests.

Everything the callbacks import from OTHER cavachon packages
(cavachon.layers.ProgressiveScaler, cavachon.dataloader.DataLoader,
cavachon.distributions.*) and every model / component / scheduler object
they receive is replaced here by lightweight fakes. Code inside
cavachon.callbacks stays real; tensorflow / numpy stay real.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import tensorflow as tf


class FakeProgressiveScaler:
    """Stand-in for cavachon.layers.ProgressiveScaler.

    Only exposes what the callbacks read: ``total_iterations``,
    ``current_iteration`` (tf.Variables, so ``float()``/``.numpy()`` work)
    and ``increment()`` which counts its calls.
    """

    def __init__(self, total_iterations=10.0, current_iteration=0.0):
        self.total_iterations = tf.Variable(float(total_iterations))
        self.current_iteration = tf.Variable(float(current_iteration))
        self.increment_calls = 0

    def increment(self):
        self.increment_calls += 1
        self.current_iteration.assign_add(1.0)


@pytest.fixture
def fake_scaler_cls(monkeypatch):
    """Patch ProgressiveScaler where annealing_callback looks it up."""
    monkeypatch.setattr(
        "cavachon.callbacks.annealing_callback.ProgressiveScaler",
        FakeProgressiveScaler,
    )
    return FakeProgressiveScaler


def make_loss(initial=0.0):
    """A loss object exposing a real tf.Variable ``weight``."""
    return SimpleNamespace(weight=tf.Variable(float(initial), dtype=tf.float32))


class FakeCavachonModel:
    """Mimics the parts of cavachon.model.Model the callbacks touch.

    Like the real model's ``__setattr__``, setting ``trainable = False``
    propagates False to every component (True does not propagate).
    """

    def __init__(self, components=None, loss=None, **extra):
        object.__setattr__(self, "components", components or {})
        object.__setattr__(self, "trainable", True)
        object.__setattr__(self, "loss", loss or {})
        for k, v in extra.items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "trainable" and not value:
            for comp in self.components.values():
                comp.trainable = value


def make_component(trainable=True, scaler=None):
    return SimpleNamespace(
        trainable=trainable,
        hierarchical_encoder=SimpleNamespace(progressive_scaler=scaler),
    )


@pytest.fixture
def make_model():
    return FakeCavachonModel


@pytest.fixture
def make_comp():
    return make_component


@pytest.fixture
def loss_factory():
    return make_loss


@pytest.fixture
def fake_scheduler():
    return MagicMock(name="scheduler", component_order=[])
