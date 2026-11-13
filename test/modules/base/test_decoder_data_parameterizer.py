"""Unit tests for cavachon.modules.base.decoder_data_parameterizer.

Mocked: ReflectionHandler.get_class_by_name (returns FakeParameterizer) and
TensorUtils.create_backbone_layers (returns a small real Sequential backbone).
"""

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.base.decoder_data_parameterizer import DecoderDataParameterizer
from test.modules.base.conftest import FakeParameterizer

MODULE = "cavachon.modules.base.decoder_data_parameterizer"


@pytest.fixture
def patched(monkeypatch, backbone_factory):
    get_class = mock.MagicMock(return_value=FakeParameterizer)
    backbone = backbone_factory(units=(6, 4))
    monkeypatch.setattr(f"{MODULE}.ReflectionHandler.get_class_by_name", get_class)
    monkeypatch.setattr(f"{MODULE}.TensorUtils.create_backbone_layers", backbone)
    return get_class, backbone


def _x(batch=2, dims=3):
    return tf.reshape(tf.range(batch * dims, dtype=tf.float32), (batch, dims)) / 10.0


class TestConstruction:
    def test_reflection_lookup_uses_distribution_name(self, patched):
        get_class, _ = patched
        DecoderDataParameterizer("IndependentZeroInflatedNegativeBinomial", name="dec")
        get_class.assert_called_once_with(
            "IndependentZeroInflatedNegativeBinomial",
            "modules/parameterizers",
            "Parameterizer",
        )

    def test_backbone_name_and_activation(self, patched):
        _, backbone = patched
        dec = DecoderDataParameterizer("Dist", name="dec")
        kwargs = backbone.call_args.kwargs
        assert kwargs["name"] == f"dec_{Constants.MODULE_BACKBONE}"
        assert kwargs["activation"] == "swish"
        assert dec.backbone_network is backbone.created[0]
        # decoder backbone is not reversed
        assert not kwargs.get("reverse", False)

    def test_parameterizer_make_arguments(self, patched):
        dec = DecoderDataParameterizer("Dist", n_vars=11, name="dec")
        assert FakeParameterizer.make_calls == [
            dict(
                input_dims=4,
                event_dims=11,
                name=f"dec_{Constants.MODULE_X_PARAMETERIZER}",
            )
        ]
        assert isinstance(dec.x_parameterizer, FakeParameterizer)

    def test_input_dims_is_last_dense_units(self, monkeypatch, backbone_factory):
        monkeypatch.setattr(
            f"{MODULE}.ReflectionHandler.get_class_by_name",
            mock.MagicMock(return_value=FakeParameterizer),
        )
        monkeypatch.setattr(
            f"{MODULE}.TensorUtils.create_backbone_layers",
            backbone_factory(units=(9, 7, 13)),
        )
        DecoderDataParameterizer("Dist", n_vars=2, name="dec")
        assert FakeParameterizer.make_calls[0]["input_dims"] == 13

    def test_input_dims_zero_without_dense(self, monkeypatch, backbone_factory):
        monkeypatch.setattr(
            f"{MODULE}.ReflectionHandler.get_class_by_name",
            mock.MagicMock(return_value=FakeParameterizer),
        )
        monkeypatch.setattr(
            f"{MODULE}.TensorUtils.create_backbone_layers", backbone_factory(units=())
        )
        DecoderDataParameterizer("Dist", n_vars=2, name="dec")
        assert FakeParameterizer.make_calls[0]["input_dims"] == 0

    def test_n_layers_forwarded_to_backbone(self, patched):
        # regression: fixed bug, see UNIT_TEST_REPORT.md
        _, backbone = patched
        DecoderDataParameterizer("Dist", n_vars=3, n_layers=5, name="dec")
        assert backbone.call_args.kwargs["n_layers"] == 5


class TestCall:
    def test_backbone_receives_matrix_and_output_shape(self, patched):
        _, backbone = patched
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        x = _x()
        out = dec(
            {Constants.TENSOR_NAME_X: x, Constants.TENSOR_NAME_LIBSIZE: tf.ones((2, 1))}
        )
        recorder = backbone.created[0].layers[0]
        np.testing.assert_allclose(recorder.calls[-1]["inputs"].numpy(), x.numpy())
        assert tuple(out.shape) == (2, 4)

    def test_parameterizer_receives_backbone_output_without_libsize(self, patched):
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        x = _x()
        out = dec(
            {
                Constants.TENSOR_NAME_X: x,
                Constants.TENSOR_NAME_LIBSIZE: 5 * tf.ones((2, 1)),
            }
        )
        expected = dec.backbone_network(x)
        last = dec.x_parameterizer.calls[-1]["inputs"]
        assert set(last) == {Constants.TENSOR_NAME_X}
        np.testing.assert_allclose(
            last[Constants.TENSOR_NAME_X].numpy(), expected.numpy(), atol=1e-6
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-6)

    def test_libsize_forwarded_when_parameterizer_scales(self, patched):
        FakeParameterizer.libsize_scaling = True
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        x = _x()
        libsize = tf.constant([[2.0], [3.0]])
        out = dec({Constants.TENSOR_NAME_X: x, Constants.TENSOR_NAME_LIBSIZE: libsize})
        last = dec.x_parameterizer.calls[-1]["inputs"]
        assert set(last) == {Constants.TENSOR_NAME_X, Constants.TENSOR_NAME_LIBSIZE}
        np.testing.assert_allclose(
            last[Constants.TENSOR_NAME_LIBSIZE].numpy(), libsize.numpy()
        )
        expected = dec.backbone_network(x).numpy() * libsize.numpy()
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    @pytest.mark.parametrize("training", [True, False])
    def test_training_flag_propagated(self, patched, training):
        _, backbone = patched
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        dec({Constants.TENSOR_NAME_X: _x()}, training=training)
        assert backbone.created[0].layers[0].calls[-1]["training"] is training
        assert dec.x_parameterizer.calls[-1]["training"] is training

    def test_compute_attribution_target(self, patched):
        _, backbone = patched
        FakeParameterizer.libsize_scaling = True
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        x = _x()
        libsize = tf.constant([[2.0], [3.0]])
        out = dec.compute_attribution_target(
            {Constants.TENSOR_NAME_X: x, Constants.TENSOR_NAME_LIBSIZE: libsize}
        )
        assert backbone.created[0].layers[0].calls[-1]["training"] is False
        att = dec.x_parameterizer.attribution_calls[-1]
        np.testing.assert_allclose(
            att[Constants.TENSOR_NAME_LIBSIZE].numpy(), libsize.numpy()
        )
        expected = dec.backbone_network(x).numpy() + 100.0
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

    def test_compute_attribution_target_no_libsize(self, patched):
        dec = DecoderDataParameterizer("Dist", n_vars=3, name="dec")
        dec.compute_attribution_target(
            {
                Constants.TENSOR_NAME_X: _x(),
                Constants.TENSOR_NAME_LIBSIZE: tf.ones((2, 1)),
            }
        )
        assert set(dec.x_parameterizer.attribution_calls[-1]) == {
            Constants.TENSOR_NAME_X
        }

    def test_train_step_not_implemented(self, patched):
        dec = DecoderDataParameterizer("Dist", name="dec")
        with pytest.raises(NotImplementedError):
            dec.train_step(None)
