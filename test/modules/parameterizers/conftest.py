"""Shared fakes for tests of cavachon.modules.parameterizers.

Parameterizer.make() looks up its layer class through
ReflectionHandler.get_class_by_name (a cavachon.utils dependency). That
lookup is patched to return FakeParameterizerLayer, a deterministic
identity layer, so the functional Keras graph is still built for real.
"""

from unittest import mock

import pytest
import tensorflow as tf

MODULE = "cavachon.modules.parameterizers.parameterizer"


class FakeParameterizerLayer(tf.keras.layers.Layer):
    """Identity 'parameterizer layer' that records its constructor args
    and every call. Being an identity, the module's post-processing
    (modify_outputs) can be checked value-by-value."""

    instances = []

    def __init__(self, event_dims, name="fake_layer", **kwargs):
        # swallow distribution-specific kwargs (e.g. n_components)
        self.extra_kwargs = dict(kwargs)
        super().__init__(name=name)
        self.event_dims = event_dims
        self.calls = []
        FakeParameterizerLayer.instances.append(self)

    def call(self, inputs):
        self.calls.append(inputs)
        return tf.identity(inputs)


@pytest.fixture
def fake_layer(monkeypatch):
    FakeParameterizerLayer.instances = []
    get_class = mock.MagicMock(return_value=FakeParameterizerLayer)
    monkeypatch.setattr(f"{MODULE}.ReflectionHandler.get_class_by_name", get_class)
    return get_class
