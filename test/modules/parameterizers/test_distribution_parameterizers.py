"""Unit tests for the concrete parameterizers in
cavachon.modules.parameterizers (Bernoulli, ZINB, MVN-diag, mixture MVN-diag,
StudentT).

Mocked: ReflectionHandler.get_class_by_name -> FakeParameterizerLayer
(identity), so that input_dims == n_params * event_dims and the output
of the module equals the post-processed input.
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.parameterizers import (
    IndependentBernoulliParameterizer,
    IndependentZeroInflatedNegativeBinomialParameterizer,
    MixtureMultivariateNormalDiagParameterizer,
    MultivariateNormalDiagParameterizer,
    Parameterizer,
    StudenttParameterizer,
)
from test.modules.parameterizers.conftest import FakeParameterizerLayer

X = Constants.TENSOR_NAME_X
L = Constants.TENSOR_NAME_LIBSIZE
ZINB = IndependentZeroInflatedNegativeBinomialParameterizer


@pytest.mark.parametrize(
    "cls, default_name, default_libsize",
    [
        (IndependentBernoulliParameterizer, "independent_bernoulli", False),
        (ZINB, "independent_zero_inflated_negative_binomial", True),
        (MultivariateNormalDiagParameterizer, "multivariate_normal_diag", False),
        (StudenttParameterizer, "student_t", False),
    ],
)
def test_reflection_name_defaults_and_class(
    fake_layer, cls, default_name, default_libsize
):
    model = cls.make(input_dims=6, event_dims=2)
    fake_layer.assert_called_once_with(cls.__name__, "layers/parameterizers", "Layer")
    assert isinstance(model, cls) and isinstance(model, Parameterizer)
    assert model.name == default_name
    assert FakeParameterizerLayer.instances[0].name == default_name
    assert FakeParameterizerLayer.instances[0].event_dims == 2
    assert cls.default_libsize_scaling is default_libsize
    assert model.libsize_scaling is default_libsize


class TestBernoulli:
    def test_logits_pass_through(self, fake_layer):
        model = IndependentBernoulliParameterizer.make(
            input_dims=3, event_dims=3, name="b"
        )
        x = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
        out = model({X: tf.constant(x)})
        np.testing.assert_allclose(out.numpy(), x)
        assert model.exp_transform is False

    def test_attribution_target_is_sigmoid(self, fake_layer):
        model = IndependentBernoulliParameterizer.make(input_dims=3, event_dims=3)
        x = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
        att = model.compute_attribution_target({X: tf.constant(x)})
        np.testing.assert_allclose(att.numpy(), 1 / (1 + np.exp(-x)), rtol=1e-6)


def _zinb_in(logits, mean, disp):
    return tf.constant(np.concatenate([logits, mean, disp], axis=-1).astype(np.float32))


class TestZINB:
    LOGITS = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    DISP = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    def test_defaults_require_libsize_input(self, fake_layer):
        model = ZINB.make(input_dims=6, event_dims=2)
        assert set(model.input) == {X, L}
        assert model.exp_transform is True

    def test_order_logits_mean_dispersion_and_transform(self, fake_layer):
        model = ZINB.make(input_dims=6, event_dims=2)
        mean = np.array([[0.5, 10.0], [0.0, -1.0]], dtype=np.float32)
        libsize = np.array([[2.0], [3.0]], dtype=np.float32)
        out = model(
            {X: _zinb_in(self.LOGITS, mean, self.DISP), L: tf.constant(libsize)}
        ).numpy()
        assert out.shape == (2, 6)
        logits, m, disp = np.split(out, 3, axis=-1)
        np.testing.assert_allclose(logits, self.LOGITS)
        np.testing.assert_allclose(disp, self.DISP)
        # libsize scaling first, then clip at 7, exp(.)-1, then floor at 1e-7
        scaled = np.minimum(mean * libsize, 7.0)
        expected = np.exp(scaled) - 1
        expected = np.where(expected <= 0, 1e-7, expected)
        np.testing.assert_allclose(m, expected, rtol=1e-5)
        assert np.all(m > 0)
        np.testing.assert_allclose(m[0, 1], np.exp(7.0) - 1, rtol=1e-5)  # clipped
        assert m[1, 0] == pytest.approx(1e-7) and m[1, 1] == pytest.approx(1e-7)

    def test_no_libsize_no_exp_only_floors_mean(self, fake_layer):
        model = ZINB.make(
            input_dims=6, event_dims=2, libsize_scaling=False, exp_transform=False
        )
        assert list(model.input) == [X]
        mean = np.array([[0.5, -3.0], [0.0, 8.0]], dtype=np.float32)
        out = model({X: _zinb_in(self.LOGITS, mean, self.DISP)}).numpy()
        _, m, _ = np.split(out, 3, axis=-1)
        np.testing.assert_allclose(m, np.where(mean <= 0, 1e-7, mean), rtol=1e-6)

    @pytest.mark.parametrize(
        "libsize_scaling, exp_transform", [(True, True), (True, False), (False, True)]
    )
    def test_attribution_target_equals_mean_slice(
        self, fake_layer, libsize_scaling, exp_transform
    ):
        model = ZINB.make(
            input_dims=6,
            event_dims=2,
            libsize_scaling=libsize_scaling,
            exp_transform=exp_transform,
        )
        mean = np.array([[0.5, 10.0], [0.0, -1.0]], dtype=np.float32)
        inputs = {X: _zinb_in(self.LOGITS, mean, self.DISP)}
        if libsize_scaling:
            inputs[L] = tf.constant([[2.0], [3.0]])
        full = model(inputs).numpy()
        att = model.compute_attribution_target(inputs).numpy()
        np.testing.assert_allclose(att, np.split(full, 3, axis=-1)[1], rtol=1e-5)


class TestMVN:
    def test_loc_scale_pass_through(self, fake_layer):
        model = MultivariateNormalDiagParameterizer.make(
            input_dims=4, event_dims=2, name="mvn"
        )
        x = np.array([[1.0, -2.0, 0.3, 0.7]], dtype=np.float32)
        out = model({X: tf.constant(x)})
        np.testing.assert_allclose(out.numpy(), x)
        assert model.exp_transform is False

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: base modify_outputs applies tf.math.exp to a KerasTensor -> make(exp_transform=True) raises",
    )
    def test_exp_transform_option(self, fake_layer):
        model = MultivariateNormalDiagParameterizer.make(
            input_dims=2, event_dims=1, exp_transform=True
        )
        x = np.array([[0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(
            model({X: tf.constant(x)}).numpy(), np.exp(x), rtol=1e-6
        )


class TestMixtureMVN:
    def test_n_components_forwarded_to_layer(self, fake_layer):
        model = MixtureMultivariateNormalDiagParameterizer.make(
            input_dims=4, event_dims=2, n_components=3
        )
        fake_layer.assert_called_once_with(
            "MixtureMultivariateNormalDiagParameterizer",
            "layers/parameterizers",
            "Layer",
        )
        layer = FakeParameterizerLayer.instances[0]
        assert layer.extra_kwargs == {"n_components": 3}
        assert layer.name == "mixture_multivariate_normal_diag"
        assert model.libsize_scaling is False and model.exp_transform is False
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        np.testing.assert_allclose(model({X: tf.constant(x)}).numpy(), x)

    def test_n_components_required(self, fake_layer):
        with pytest.raises(TypeError):
            MixtureMultivariateNormalDiagParameterizer.make(input_dims=4, event_dims=2)


class TestStudentT:
    def test_pass_through(self, fake_layer):
        model = StudenttParameterizer.make(input_dims=6, event_dims=2, name="t")
        x = np.arange(6, dtype=np.float32).reshape(1, 6)
        np.testing.assert_allclose(model({X: tf.constant(x)}).numpy(), x)
        assert list(model.input) == [X]

    def test_scaling_options_are_forced_off(self, fake_layer):
        # current (documented-by-comment) behaviour: options are overridden
        model = StudenttParameterizer.make(
            input_dims=6, event_dims=2, libsize_scaling=True, exp_transform=True
        )
        assert model.libsize_scaling is False and model.exp_transform is False
        assert FakeParameterizerLayer.instances[0].extra_kwargs == {}
