"""Shared fakes for tests of cavachon.modules.preprocessors.

The modifier models in cavachon.modules.preprocessors.modifiers use layers
from cavachon.layers.modifiers (ToDense, LogTransform, NormalizeLibrarySize,
Binarize). Those are replaced by tiny dict-in/dict-out Keras layers that
apply distinct, non-commuting arithmetic so the application order can be
read off the output values:

    ToDense  : x + 1
    LogTransform : x * 2
    NormalizeLibrarySize : x - 3, and adds '<key>_libsize' = row sum of x
    Binarize : x * 10

Each fake appends (class name, key) to a shared call log.
"""

from unittest import mock

import pytest
import tensorflow as tf

from cavachon.modules.preprocessors import modifiers as M

MODIFIER_MODULES = {
    "bernoulli": "cavachon.modules.preprocessors.modifiers.independent_bernoulli_modifier",
    "zinb": "cavachon.modules.preprocessors.modifiers.independent_zero_inflated_negative_binomial_modifier",
    "mvn": "cavachon.modules.preprocessors.modifiers.multivariate_normal_diag_modifier",
    "studentt": "cavachon.modules.preprocessors.modifiers.studentt_modifier",
}

CALL_LOG = []


class _KeyModifier(tf.keras.layers.Layer):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def op(self, x):
        raise NotImplementedError

    def call(self, inputs):
        CALL_LOG.append((type(self).__name__, self.key))
        out = dict(inputs)
        out[self.key] = self.op(inputs[self.key])
        return out


class FakeToDense(_KeyModifier):
    def op(self, x):
        return x + 1.0


class FakeLogTransform(_KeyModifier):
    def op(self, x):
        return x * 2.0


class FakeBinarize(_KeyModifier):
    def op(self, x):
        return x * 10.0


class FakeNormalizeLibrarySize(_KeyModifier):
    def op(self, x):
        return x - 3.0

    def call(self, inputs):
        out = super().call(inputs)
        out[f"{self.key}_libsize"] = tf.reduce_sum(
            inputs[self.key], axis=-1, keepdims=True
        )
        return out


@pytest.fixture
def fake_modifier_layers(monkeypatch):
    CALL_LOG.clear()
    for path in MODIFIER_MODULES.values():
        monkeypatch.setattr(f"{path}.ToDense", FakeToDense)
    monkeypatch.setattr(f"{MODIFIER_MODULES['bernoulli']}.Binarize", FakeBinarize)
    monkeypatch.setattr(f"{MODIFIER_MODULES['zinb']}.LogTransform", FakeLogTransform)
    monkeypatch.setattr(
        f"{MODIFIER_MODULES['zinb']}.NormalizeLibrarySize", FakeNormalizeLibrarySize
    )
    return CALL_LOG


REAL_MODIFIERS = {
    "IndependentBernoulli": M.IndependentBernoulliModifier,
    "IndependentZeroInflatedNegativeBinomial": M.IndependentZeroInflatedNegativeBinomialModifier,
    "MultivariateNormalDiag": M.MultivariateNormalDiagModifier,
    "Studentt": M.StudenttModifier,
}


@pytest.fixture
def modifier_lookup(monkeypatch, fake_modifier_layers):
    """Patch ReflectionHandler.get_class_by_name as seen by preprocessor.py so
    it returns the real in-package modifier class for a distribution name."""
    get_class = mock.MagicMock(
        side_effect=lambda name, subdir, postfix: REAL_MODIFIERS[name]
    )
    monkeypatch.setattr(
        "cavachon.modules.preprocessors.preprocessor.ReflectionHandler.get_class_by_name",
        get_class,
    )
    return get_class
