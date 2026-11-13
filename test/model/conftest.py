"""Shared fakes for the ``cavachon.model`` unit tests.

Everything ``cavachon/model/model.py`` imports from other cavachon packages
is replaced, at the place model.py looks it up (``cavachon.model.model.X``),
by small recording fakes:

* ``Component``          -> ``FakeComponentFactory`` (``make`` builds a
                            ``FakeComponent``, a tiny real ``tf.keras.Model``)
* ``GeneralUtils``       -> ``MagicMock`` (``order_components`` = identity)
* ``TensorUtils``        -> ``MagicMock`` (``remove_nan_gradients`` = identity)
* ``DataLoader``         -> ``MagicMock`` (tests set ``return_value``)
* ``ToDense``            -> ``FakeToDense``
* ``ProgressiveScaler``  -> ``FakeScaler``
* ``StandardKLDivergence`` / ``GMMKLDivergence`` /
  ``NegativeLogDataLikelihood`` / ``GMMDensityLoss`` -> recording fake losses

``ComponentConfig`` is replaced by ``FakeConfig`` instances (attributes +
``model_dump()``). ``cavachon.environment.Constants`` stays real.

FakeComponent numerics (deterministic, per modality ``m``):

    z                     = sum_m x_m @ W_m      (W_m trainable, const init)
    z_parameters          = 2 * z
    z_prior_parameters    = 0 * z + 7
    z_hat                 = z + z_hat_offset
    z_hat_parameters      = 3 * z_hat
    z_hat_prior_parameters= 0 * z_hat + 11
    {m}_x_parameters      = rowsum(z_hat) * ones(n_vars[m])
    z_conditional_echo    = identity(inputs['z_conditional'])      (if given)
    z_hat_conditional_echo= identity(inputs['z_hat_conditional'])  (if given)

Fake losses compute ``factor * reduce_sum(y_pred)`` where ``factor`` is the
output of their (fake) weight.
"""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

import cavachon.model.model as model_module
from cavachon.environment.constants import Constants


# ---------------------------------------------------------------------------
# Config stand-in
# ---------------------------------------------------------------------------


class FakeConfig:
    """Stand-in for ComponentConfig: plain attributes + model_dump()."""

    def __init__(
        self,
        name,
        modality_names=("rna",),
        n_vars=None,
        n_vars_batch_effect=None,
        distribution_names=None,
        conditioned_on_z=(),
        conditioned_on_z_hat=(),
        save_x=None,
        save_z=None,
        z_hat_kl_weight=None,
        latent_dim=2,
        kernel_value=0.5,
        z_hat_offset=10.0,
        learn_z_hat_priors=False,
        has_z_hat_prior_parameterizer=True,
        kernel_regularizer=None,
    ):
        modality_names = list(modality_names)
        self.name = name
        self.modality_names = modality_names
        self.n_vars = dict(n_vars) if n_vars else {m: 2 for m in modality_names}
        self.n_vars_batch_effect = (
            dict(n_vars_batch_effect)
            if n_vars_batch_effect
            else {m: 1 for m in modality_names}
        )
        self.distribution_names = (
            dict(distribution_names)
            if distribution_names
            else {m: f"dist_{m}" for m in modality_names}
        )
        self.conditioned_on_z = list(conditioned_on_z)
        self.conditioned_on_z_hat = list(conditioned_on_z_hat)
        self.save_x = dict(save_x) if save_x is not None else {m: True for m in modality_names}
        self.save_z = dict(save_z) if save_z is not None else {m: True for m in modality_names}
        self.z_hat_kl_weight = z_hat_kl_weight
        self.latent_dim = latent_dim
        self.kernel_value = kernel_value
        self.z_hat_offset = z_hat_offset
        self.learn_z_hat_priors = learn_z_hat_priors
        self.has_z_hat_prior_parameterizer = has_z_hat_prior_parameterizer
        self.kernel_regularizer = kernel_regularizer

    def model_dump(self):
        return {k: copy.copy(v) for k, v in vars(self).items()}


# ---------------------------------------------------------------------------
# Component stand-in
# ---------------------------------------------------------------------------


class Recorder:
    """Plain (non Keras-tracked) holder for call records."""

    def __init__(self):
        self.calls = []
        self.encode_calls = []
        self.hierarchical_encode_calls = []
        self.decode_calls = []


class FakeComponent(tf.keras.Model):
    def __init__(
        self,
        name,
        modality_names,
        n_vars,
        latent_dim=2,
        kernel_value=0.5,
        z_hat_offset=10.0,
        learn_z_hat_priors=False,
        has_z_hat_prior_parameterizer=True,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.modality_names = list(modality_names)
        self.n_vars_map = dict(n_vars)
        self.latent_dim = latent_dim
        self.z_hat_offset = float(z_hat_offset)
        self.encoders = {
            m: tf.keras.layers.Dense(
                latent_dim,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(kernel_value),
                kernel_regularizer=kernel_regularizer,
                name=f"{name}_{m}_encoder",
            )
            for m in self.modality_names
        }
        self.z_prior_parameterizer = SimpleNamespace(event_dims=latent_dim)
        self.learn_z_hat_priors = learn_z_hat_priors
        self.z_hat_prior_parameterizer = (
            SimpleNamespace(event_dims=latent_dim)
            if has_z_hat_prior_parameterizer
            else None
        )
        self.recorder = Recorder()

    # -- numerics ---------------------------------------------------------
    def _z(self, batch):
        return tf.add_n(
            [
                self.encoders[m](batch[f"{m}_{Constants.TENSOR_NAME_X}"])
                for m in self.modality_names
            ]
        )

    def _x_params(self, z_hat):
        return {
            f"{m}_{Constants.MODEL_OUTPUTS_X_PARAMS}": tf.reduce_sum(
                z_hat, axis=-1, keepdims=True
            )
            * tf.ones((1, self.n_vars_map[m]))
            for m in self.modality_names
        }

    def call(self, inputs, training=None):
        self.recorder.calls.append({"keys": sorted(inputs.keys()), "training": training})
        z = self._z(inputs)
        z_hat = z + self.z_hat_offset
        outputs = {
            Constants.MODEL_OUTPUTS_Z: z,
            Constants.MODEL_OUTPUTS_Z_HAT: z_hat,
            Constants.MODEL_OUTPUTS_Z_PARAMS: 2.0 * z,
            Constants.MODEL_OUTPUTS_Z_PRIOR_PARAMS: 0.0 * z + 7.0,
            Constants.MODEL_OUTPUTS_Z_HAT_PARAMS: 3.0 * z_hat,
            Constants.MODEL_OUTPUTS_Z_HAT_PRIOR_PARAMS: 0.0 * z_hat + 11.0,
        }
        outputs.update(self._x_params(z_hat))
        if Constants.MODULE_INPUTS_CONDITIONED_Z in inputs:
            outputs["z_conditional_echo"] = tf.identity(
                inputs[Constants.MODULE_INPUTS_CONDITIONED_Z]
            )
        if Constants.MODULE_INPUTS_CONDITIONED_Z_HAT in inputs:
            outputs["z_hat_conditional_echo"] = tf.identity(
                inputs[Constants.MODULE_INPUTS_CONDITIONED_Z_HAT]
            )
        return outputs

    # -- explicit sub-paths -------------------------------------------------
    def encode(self, batch, training=False):
        self.recorder.encode_calls.append({"batch": dict(batch), "training": training})
        z = self._z(batch)
        return {Constants.MODEL_OUTPUTS_Z_PARAMS: 2.0 * z, Constants.MODEL_OUTPUTS_Z: z}

    def hierarchical_encode(self, batch, z, training=False):
        self.recorder.hierarchical_encode_calls.append(
            {"batch": dict(batch), "z": z, "training": training}
        )
        return {Constants.MODEL_OUTPUTS_Z_HAT: z + self.z_hat_offset}

    def decode(self, batch, z_hat, training=False):
        self.recorder.decode_calls.append(
            {"batch": dict(batch), "z_hat": z_hat, "training": training}
        )
        return self._x_params(z_hat)


class FakeComponentFactory:
    """Replaces ``cavachon.model.model.Component``; only ``make`` is used."""

    def __init__(self):
        self.make_calls = []

    def make(self, **kwargs):
        self.make_calls.append(dict(kwargs))
        return FakeComponent(**kwargs)


# ---------------------------------------------------------------------------
# Layers stand-ins
# ---------------------------------------------------------------------------


class FakeScaler:
    """Stand-in for ProgressiveScaler: ``__call__(x) = factor * x``."""

    def __init__(self, total_iterations=5000, scale=1.0, name="progressive_scaler"):
        self.total_iterations = total_iterations
        self.scale = float(scale)
        self.factor = float(scale)
        self.name = name
        self.pins = []
        self.activations = []
        self.n_calls = 0

    def __call__(self, inputs, training=False):
        self.n_calls += 1
        return tf.constant(self.factor, dtype=tf.float32) * tf.cast(inputs, tf.float32)

    def pin_to(self, value):
        self.pins.append(float(value))
        self.factor = float(value)

    def activate(self, total_iterations):
        self.activations.append(float(total_iterations))
        self.factor = 0.0


class FakeToDense:
    """Stand-in for ToDense; returns a new mapping with ``key`` replaced by
    ``transform(value)``. ``transform`` can be overridden per test."""

    instances = []
    transform = staticmethod(lambda v: v)

    def __init__(self, key, *args, **kwargs):
        self.key = key
        FakeToDense.instances.append(self)

    def __call__(self, inputs):
        out = dict(inputs)
        out[self.key] = FakeToDense.transform(inputs[self.key])
        return out


# ---------------------------------------------------------------------------
# Loss stand-ins
# ---------------------------------------------------------------------------


class _FakeScaledLoss:
    instances = None  # set per subclass

    def __init__(self, weight=1.0, name=None):
        self.weight = FakeScaler(total_iterations=1, scale=float(weight))
        self.init_weight = weight
        self.name = name
        self.calls = []
        type(self).instances.append(self)

    def raw(self, y_true, y_pred):
        return tf.reduce_sum(y_pred)

    def __call__(self, y_true, y_pred):
        self.calls.append((y_true, y_pred))
        return self.weight(tf.ones(())) * self.raw(y_true, y_pred)


class FakeStandardKL(_FakeScaledLoss):
    instances = []


class FakeGMMKL(_FakeScaledLoss):
    instances = []


class FakeNLL(_FakeScaledLoss):
    instances = []

    def __init__(self, distribution_name, weight=1.0, name=None):
        self.distribution_name = distribution_name
        super().__init__(weight=weight, name=name)


class FakeGMMDensity:
    instances = []

    def __init__(self, weight=None, name=None):
        self.weight = weight
        self.name = name
        self.calls = []
        FakeGMMDensity.instances.append(self)

    def __call__(self, y_true, y_pred):
        self.calls.append((y_true, y_pred))
        return self.weight * tf.reduce_sum(y_pred)


class PlainLoss:
    """A custom loss with no ``weight`` attribute."""

    def __init__(self, factor=1.0):
        self.factor = factor
        self.calls = []

    def __call__(self, y_true, y_pred):
        self.calls.append((y_true, y_pred))
        return self.factor * tf.reduce_sum(y_pred)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    """Patch every cross-package dependency of cavachon.model.model."""
    for cls in (FakeStandardKL, FakeGMMKL, FakeNLL, FakeGMMDensity):
        cls.instances.clear()
    FakeToDense.instances.clear()
    FakeToDense.transform = staticmethod(lambda v: v)

    ns = SimpleNamespace()
    ns.component = FakeComponentFactory()
    ns.general_utils = MagicMock(name="GeneralUtils")
    ns.general_utils.order_components.side_effect = lambda cfgs: list(cfgs)
    ns.tensor_utils = MagicMock(name="TensorUtils")
    ns.tensor_utils.remove_nan_gradients.side_effect = lambda grads: list(grads)
    ns.dataloader = MagicMock(name="DataLoader")

    monkeypatch.setattr(model_module, "Component", ns.component)
    monkeypatch.setattr(model_module, "GeneralUtils", ns.general_utils)
    monkeypatch.setattr(model_module, "TensorUtils", ns.tensor_utils)
    monkeypatch.setattr(model_module, "DataLoader", ns.dataloader)
    monkeypatch.setattr(model_module, "ToDense", FakeToDense)
    monkeypatch.setattr(model_module, "ProgressiveScaler", FakeScaler)
    monkeypatch.setattr(model_module, "StandardKLDivergence", FakeStandardKL)
    monkeypatch.setattr(model_module, "GMMKLDivergence", FakeGMMKL)
    monkeypatch.setattr(model_module, "NegativeLogDataLikelihood", FakeNLL)
    monkeypatch.setattr(model_module, "GMMDensityLoss", FakeGMMDensity)
    return ns


@pytest.fixture
def make_model(fakes):
    """Build a Model via Model.make with the fakes in place."""

    def _make(*configs, name="cavachon"):
        tf.keras.backend.clear_session()
        return model_module.Model.make(list(configs), name=name)

    return _make


def make_batch(model_or_modalities, n_obs=3, seed=0, n_vars=None):
    """Deterministic batch dict for the given modalities ({mod: n_vars})."""
    rng = np.random.default_rng(seed)
    batch = {}
    for m, nv in model_or_modalities.items():
        batch[f"{m}_{Constants.TENSOR_NAME_X}"] = tf.constant(
            rng.integers(0, 5, size=(n_obs, nv)).astype(np.float32)
        )
        batch[f"{m}_{Constants.TENSOR_NAME_BATCH}"] = tf.zeros((n_obs, 1))
    return batch
