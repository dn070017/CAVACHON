"""Unit tests for cavachon.modules.parameterizers.parameterizer.Parameterizer.

Mocked: ReflectionHandler.get_class_by_name -> FakeParameterizerLayer
(identity). Everything else (functional graph, modify_outputs) is real.
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.parameterizers.parameterizer import Parameterizer
from test.modules.parameterizers.conftest import FakeParameterizerLayer

X = Constants.TENSOR_NAME_X
L = Constants.TENSOR_NAME_LIBSIZE

X_VAL = np.array([[0.5, -1.0, 2.0, 0.0], [1.0, 3.0, -2.0, 0.25]], dtype=np.float32)
L_VAL = np.array([[2.0], [0.5]], dtype=np.float32)


class TestSetupInputs:
    def test_without_libsize(self):
        inputs = Parameterizer.setup_inputs(7)
        assert list(inputs) == [X]
        assert tuple(inputs[X].shape) == (None, 7)

    def test_with_libsize(self):
        inputs = Parameterizer.setup_inputs(7, libsize_scaling=True, extra="ignored")
        assert list(inputs) == [X, L]
        assert tuple(inputs[L].shape) == (None, 1)


class TestModifyOutputs:
    def test_identity_by_default(self):
        out = Parameterizer.modify_outputs({X: None}, tf.constant(X_VAL))
        np.testing.assert_allclose(out.numpy(), X_VAL)

    def test_libsize_scaling(self):
        out = Parameterizer.modify_outputs(
            {L: tf.constant(L_VAL)}, tf.constant(X_VAL), libsize_scaling=True
        )
        np.testing.assert_allclose(out.numpy(), X_VAL * L_VAL)

    def test_scale_then_exp(self):
        out = Parameterizer.modify_outputs(
            {L: tf.constant(L_VAL)},
            tf.constant(X_VAL),
            libsize_scaling=True,
            exp_transform=True,
        )
        np.testing.assert_allclose(out.numpy(), np.exp(X_VAL * L_VAL), rtol=1e-6)


class TestMake:
    def test_reflection_and_layer_arguments(self, fake_layer):
        model = Parameterizer.make(input_dims=4, event_dims=2, name="param")
        fake_layer.assert_called_once_with(
            "Parameterizer", "layers/parameterizers", "Layer"
        )
        layer = FakeParameterizerLayer.instances[0]
        assert layer.event_dims == 2
        assert layer.name == "param"
        assert model.layer is layer
        assert model.name == "param"
        assert model.libsize_scaling is False and model.exp_transform is False

    def test_extra_kwargs_reach_layer(self, fake_layer):
        Parameterizer.make(input_dims=4, event_dims=2, name="p", foo=3)
        assert FakeParameterizerLayer.instances[0].extra_kwargs == {"foo": 3}

    def test_call_without_libsize(self, fake_layer):
        model = Parameterizer.make(input_dims=4, event_dims=2)
        assert list(model.input) == [X]
        out = model({X: tf.constant(X_VAL)})
        np.testing.assert_allclose(out.numpy(), X_VAL)

    def test_call_with_libsize(self, fake_layer):
        model = Parameterizer.make(input_dims=4, event_dims=2, libsize_scaling=True)
        assert set(model.input) == {X, L}
        out = model({X: tf.constant(X_VAL), L: tf.constant(L_VAL)})
        np.testing.assert_allclose(out.numpy(), X_VAL * L_VAL)
        assert model.libsize_scaling is True and model.exp_transform is False

    def test_make_with_exp_transform(self, fake_layer):
        model = Parameterizer.make(
            input_dims=4, event_dims=2, libsize_scaling=True, exp_transform=True
        )
        out = model({X: tf.constant(X_VAL), L: tf.constant(L_VAL)})
        np.testing.assert_allclose(out.numpy(), np.exp(X_VAL * L_VAL), rtol=1e-6)

    def test_compute_attribution_target_matches_call(self, fake_layer):
        model = Parameterizer.make(input_dims=4, event_dims=2, libsize_scaling=True)
        inputs = {X: tf.constant(X_VAL), L: tf.constant(L_VAL)}
        np.testing.assert_allclose(
            model.compute_attribution_target(inputs).numpy(),
            model(inputs).numpy(),
            rtol=1e-6,
        )
        # attribution path goes through the same layer object
        np.testing.assert_allclose(
            FakeParameterizerLayer.instances[0].calls[-1].numpy(), X_VAL
        )

    def test_compute_attribution_target_applies_exp_eagerly(self, fake_layer):
        model = Parameterizer.make(input_dims=4, event_dims=2, libsize_scaling=True)
        model.exp_transform = True  # eager path of modify_outputs supports exp
        inputs = {X: tf.constant(X_VAL), L: tf.constant(L_VAL)}
        np.testing.assert_allclose(
            model.compute_attribution_target(inputs).numpy(),
            np.exp(X_VAL * L_VAL),
            rtol=1e-6,
        )
