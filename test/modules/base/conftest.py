"""Shared fakes for tests of cavachon.modules.base.

Everything imported by cavachon.modules.base from other cavachon packages
(cavachon.layers.*, cavachon.utils.TensorUtils, cavachon.utils.ReflectionHandler)
is replaced here by tiny, deterministic tf.keras layers so that the Keras
build/call machinery still runs for real.
"""

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf


class RecordingLayer(tf.keras.layers.Layer):
    """Identity layer that records every call (inputs, training, mask)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def call(self, inputs, training=None, mask=None):
        self.calls.append(dict(inputs=inputs, training=training, mask=mask))
        return tf.identity(inputs)

    def compute_mask(self, inputs, mask=None):
        return None


class FakeMVNLayer(tf.keras.layers.Layer):
    """Stand-in for MultivariateNormalDiagParameterizerLayer.

    Output is concat([x @ loc_w + loc_b, softplus(x @ scale_w + scale_b)]),
    with loc_w = 1, loc_b = 0, scale_w = 0, scale_b = 0 at build time.
    Records constructor args and every call.
    """

    instances = []

    def __init__(self, event_dims, name="fake_mvn", **kwargs):
        super().__init__(name=name, **kwargs)
        self.event_dims = event_dims
        self.init_args = dict(event_dims=event_dims, name=name, **kwargs)
        self.calls = []
        FakeMVNLayer.instances.append(self)

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        self.loc_weight = self.add_weight(
            name="loc_weight", shape=(d_in, self.event_dims), initializer="ones"
        )
        self.loc_bias = self.add_weight(
            name="loc_bias", shape=(1, self.event_dims), initializer="zeros"
        )
        self.scale_diag_weight = self.add_weight(
            name="scale_diag_weight", shape=(d_in, self.event_dims), initializer="zeros"
        )
        self.scale_diag_bias = self.add_weight(
            name="scale_diag_bias", shape=(1, self.event_dims), initializer="zeros"
        )

    def call(self, inputs, training=None, mask=None):
        self.calls.append(dict(inputs=inputs, training=training, mask=mask))
        loc = tf.matmul(inputs, self.loc_weight) + self.loc_bias
        scale = tf.math.softplus(
            tf.matmul(inputs, self.scale_diag_weight) + self.scale_diag_bias
        )
        return tf.concat([loc, scale], axis=-1)


class FakeProgressiveScaler(tf.keras.layers.Layer):
    """Stand-in for ProgressiveScaler: multiplies by a fixed 0.5."""

    FACTOR = 0.5
    instances = []

    def __init__(self, total_iterations=5000, **kwargs):
        super().__init__(**kwargs)
        self.total_iterations_arg = total_iterations
        self.calls = []
        FakeProgressiveScaler.instances.append(self)

    def call(self, inputs, training=None):
        self.calls.append(dict(inputs=inputs, training=training))
        return inputs * self.FACTOR


class FakeParameterizer(tf.keras.layers.Layer):
    """Stand-in for a modules.parameterizers class found by reflection.

    call(): returns inputs['matrix'] * inputs['libsize'] (when present).
    compute_attribution_target(): returns inputs['matrix'] + 100.
    """

    libsize_scaling = False
    make_calls = []

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name"))
        self.make_kwargs = kwargs
        self.calls = []
        self.attribution_calls = []

    @classmethod
    def make(cls, **kwargs):
        cls.make_calls.append(kwargs)
        return cls(**kwargs)

    def call(self, inputs, training=None, mask=None):
        self.calls.append(dict(inputs=dict(inputs), training=training, mask=mask))
        out = inputs["matrix"]
        if "libsize" in inputs:
            out = out * inputs["libsize"]
        return out

    def compute_attribution_target(self, inputs):
        self.attribution_calls.append(dict(inputs))
        return inputs["matrix"] + 100.0


def make_backbone(units=(6, 4), name="backbone"):
    """A real Sequential [Recording, Dense(u0), LayerNorm, Dense(u1), ...]."""
    layers = [RecordingLayer(name=f"{name}_recorder")]
    for i, u in enumerate(units):
        layers.append(
            tf.keras.layers.Dense(
                u,
                kernel_initializer="ones",
                bias_initializer="zeros",
                name=f"{name}_dense_{i}",
            )
        )
        layers.append(tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}"))
    return tf.keras.Sequential(layers, name=name)


@pytest.fixture(autouse=True)
def _reset_fake_registries():
    FakeMVNLayer.instances = []
    FakeProgressiveScaler.instances = []
    FakeParameterizer.make_calls = []
    FakeParameterizer.libsize_scaling = False
    yield


@pytest.fixture
def backbone_factory(request):
    """MagicMock replacing TensorUtils.create_backbone_layers; returns a
    fresh fake backbone (Dense widths given by `units`) per call."""

    def _factory(units=(6, 4)):
        created = []

        def side_effect(*args, **kwargs):
            bb = make_backbone(units, name=kwargs.get("name", "backbone"))
            created.append(bb)
            return bb

        m = mock.MagicMock(side_effect=side_effect)
        m.created = created
        return m

    return _factory


def np_(t):
    return np.asarray(t)
