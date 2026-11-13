"""Unit tests for cavachon.modules.base.encoder_latent_parameterizer.

Mocked: TensorUtils.create_backbone_layers, TensorUtils.max_n_neurons and
MultivariateNormalDiagParameterizerLayer (replaced by FakeMVNLayer).
"""

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.base.encoder_latent_parameterizer import (
    EncoderLatentParameterizer,
)
from test.modules.base.conftest import FakeMVNLayer

MODULE = "cavachon.modules.base.encoder_latent_parameterizer"


@pytest.fixture
def patched(monkeypatch, backbone_factory):
    backbone = backbone_factory(units=(6, 4))
    max_n = mock.MagicMock(return_value=77)
    monkeypatch.setattr(f"{MODULE}.TensorUtils.create_backbone_layers", backbone)
    monkeypatch.setattr(f"{MODULE}.TensorUtils.max_n_neurons", max_n)
    monkeypatch.setattr(
        f"{MODULE}.MultivariateNormalDiagParameterizerLayer", FakeMVNLayer
    )
    return backbone, max_n


class TestConstruction:
    def test_backbone_factory_arguments(self, patched):
        backbone, _ = patched
        enc = EncoderLatentParameterizer(n_layers=4, n_latent_dims=3, name="enc")
        backbone.assert_called_once_with(
            4, reverse=True, name=f"enc_{Constants.MODULE_BACKBONE}"
        )
        assert enc.backbone_network is backbone.created[0]

    def test_z_parameterizer_arguments(self, patched):
        enc = EncoderLatentParameterizer(n_layers=2, n_latent_dims=3, name="enc")
        assert len(FakeMVNLayer.instances) == 1
        assert enc.z_parameterizer is FakeMVNLayer.instances[0]
        assert FakeMVNLayer.instances[0].init_args == dict(
            event_dims=3, name=f"enc_{Constants.MODULE_Z_PARAMETERIZER}"
        )

    def test_max_n_neurons_from_backbone_layers(self, patched):
        backbone, max_n = patched
        enc = EncoderLatentParameterizer(n_layers=2, n_latent_dims=3, name="enc")
        max_n.assert_called_once()
        assert list(max_n.call_args.args[0]) == list(backbone.created[0].layers)
        assert enc.max_n_neurons == 77

    def test_defaults(self, patched):
        backbone, _ = patched
        EncoderLatentParameterizer(name="enc")
        assert backbone.call_args.args == (3,)
        assert FakeMVNLayer.instances[0].event_dims == 5


class TestCall:
    def test_wiring_and_values(self, patched):
        backbone, _ = patched
        enc = EncoderLatentParameterizer(n_layers=2, n_latent_dims=3, name="enc")
        x = tf.constant([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]])
        out = enc(x)
        recorder = backbone.created[0].layers[0]
        np.testing.assert_allclose(recorder.calls[-1]["inputs"].numpy(), x.numpy())
        hidden = enc.backbone_network(x)
        np.testing.assert_allclose(
            FakeMVNLayer.instances[0].calls[-1]["inputs"].numpy(),
            hidden.numpy(),
            atol=1e-6,
        )
        # (batch, 2 * n_latent_dims): loc = sum(hidden), scale = softplus(0)
        assert tuple(out.shape) == (2, 6)
        expected_loc = np.repeat(hidden.numpy().sum(-1, keepdims=True), 3, axis=-1)
        np.testing.assert_allclose(out.numpy()[:, :3], expected_loc, atol=1e-5)
        np.testing.assert_allclose(out.numpy()[:, 3:], np.log(2.0), atol=1e-6)

    @pytest.mark.parametrize("training", [True, False])
    def test_training_flag_propagated(self, patched, training):
        backbone, _ = patched
        enc = EncoderLatentParameterizer(n_layers=2, n_latent_dims=3, name="enc")
        enc(tf.ones((2, 3)), training=training)
        assert backbone.created[0].layers[0].calls[-1]["training"] is training
        assert FakeMVNLayer.instances[0].calls[-1]["training"] is training

    def test_train_step_not_implemented(self, patched):
        enc = EncoderLatentParameterizer(name="enc")
        with pytest.raises(NotImplementedError):
            enc.train_step(None)


class TestConfig:
    @pytest.mark.xfail(
        strict=True,
        reason="BUG: no get_config; Keras 3 auto-config records default n_layers/n_latent_dims",
    )
    def test_get_config_records_constructor_arguments(self, patched):
        enc = EncoderLatentParameterizer(n_layers=2, n_latent_dims=4, name="enc")
        config = enc.get_config()
        assert config["n_layers"] == 2
        assert config["n_latent_dims"] == 4

    def test_from_config_accepts_keras_base_keys(self, patched):
        config = dict(
            name="enc2", n_layers=2, n_latent_dims=3, trainable=False, dtype="float32"
        )
        enc = EncoderLatentParameterizer.from_config(config)
        assert enc.name == "enc2"
        assert enc.trainable is False
