"""Shared fakes for cavachon.modules.components unit tests.

``component.py`` reaches outside the ``cavachon.modules`` package through
the names below. Every one of them is replaced in the
``cavachon.modules.components.component`` namespace (where component.py
looks them up) by the ``fakes`` fixture:

- cavachon.layers: ToDense, ProgressiveScaler,
  MixtureMultivariateNormalDiagParameterizerLayer,
  MultivariateNormalDiagSampler
- cavachon.losses: GMMDensityLoss, GMMKLDivergence,
  NegativeLogDataLikelihood, StandardKLDivergence

The ``cavachon.modules`` builders (Preprocessor, EncoderLatentParameterizer,
HierarchicalEncoder, DecoderDataParameterizer) are also replaced by tiny
deterministic fakes so that a test controls every number. The real
HierarchicalEncoder is exercised in one file, with its own
``cavachon.layers`` dependencies patched in the modules.base namespace
(see ``real_hierarchical_encoder_env``).

Constants (cavachon.environment.Constants) are used as-is.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

import cavachon.modules.base.hierarchical_encoder as he_module
import cavachon.modules.components.component as component_module
from cavachon.environment.constants import Constants

# Tiny dimensions used throughout.
BATCH = 4
N_LATENT = 2
N_PRIORS = 2
MODALITIES = ["rna", "atac"]
DISTRIBUTIONS = {"rna": "dist_rna", "atac": "dist_atac"}
N_VARS = {"rna": 6, "atac": 5}
N_BATCH_EFFECT = {"rna": 2, "atac": 3}
# Preprocessor fake emits a libsize tensor only for these modalities.
LIBSIZE_MODALITIES = {"rna"}
# The fake z_hat sampler (stochastic=False) adds this shift to the loc,
# so z_hat is always distinguishable from z.
Z_HAT_SHIFT = 100.0


def _det_matrix(rows, cols, start=0.05, step=0.01):
    """Deterministic, non-degenerate weight matrix."""
    values = start + step * np.arange(rows * cols, dtype=np.float32)
    return values.reshape(rows, cols)


# ---------------------------------------------------------------------------
# cavachon.modules fakes
# ---------------------------------------------------------------------------
class FakePreprocessor(tf.keras.layers.Layer):
    """Concatenates the modality matrices (in modality order) under
    'matrix'; emits `{m}_matrix_libsize` = row sums for LIBSIZE_MODALITIES.
    """

    instances = []

    def __init__(self, modality_names, distribution_names, n_vars, n_dims, name):
        super().__init__(name=name)
        self.modality_names = list(modality_names)
        self.distribution_names = dict(distribution_names)
        self.n_vars = dict(n_vars)
        self.n_dims = n_dims
        self.matrix_key = Constants.TENSOR_NAME_X
        self.training_calls = []
        self.input_keys = []
        FakePreprocessor.instances.append(self)

    @classmethod
    def make(
        cls,
        modality_names,
        distribution_names,
        n_vars,
        n_dims=1024,
        name="preprocessor",
        **kwargs,
    ):
        return cls(modality_names, distribution_names, n_vars, n_dims, name)

    def call(self, inputs, training=None):
        self.training_calls.append(training)
        self.input_keys.append(sorted(inputs.keys()))
        outputs = {}
        parts = []
        for m in self.modality_names:
            x = inputs[f"{m}_{Constants.TENSOR_NAME_X}"]
            parts.append(x)
            if m in LIBSIZE_MODALITIES:
                outputs[
                    f"{m}_{Constants.TENSOR_NAME_X}_{Constants.TENSOR_NAME_LIBSIZE}"
                ] = tf.reduce_sum(x, axis=-1, keepdims=True)
        outputs[self.matrix_key] = tf.concat(parts, axis=-1)
        return outputs


class FakeEncoder(tf.keras.layers.Layer):
    """x @ W -> (batch, 2 * n_latent_dims): [loc, scale]."""

    instances = []

    def __init__(self, n_layers=3, n_latent_dims=5, name="encoder", **kwargs):
        super().__init__(name=name)
        self.n_layers = n_layers
        self.n_latent_dims = n_latent_dims
        self.max_n_neurons = 7
        self.training_calls = []
        FakeEncoder.instances.append(self)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 2 * self.n_latent_dims),
            initializer=tf.keras.initializers.Constant(
                _det_matrix(input_shape[-1], 2 * self.n_latent_dims)
            ),
            name="w",
        )

    def call(self, inputs, training=None):
        self.training_calls.append(training)
        return tf.matmul(inputs, self.w)


class FakeHierarchicalEncoder(tf.keras.layers.Layer):
    """Mirrors HierarchicalEncoder's constructor signature; concatenates
    [z_conditional?, z_hat_conditional?, z] and maps it to
    (batch, 2 * n_latent_dims). Records the inputs it was called with."""

    instances = []

    def __init__(
        self,
        n_latent_dims=5,
        is_conditioned_on_z=False,
        is_conditioned_on_z_hat=False,
        progressive_iterations=5000,
        use_bias=False,
        name="hierarchical_encoder",
        **kwargs,
    ):
        super().__init__(name=name)
        self.n_latent_dims = n_latent_dims
        self.is_conditioned_on_z = is_conditioned_on_z
        self.is_conditioned_on_z_hat = is_conditioned_on_z_hat
        self.progressive_iterations = progressive_iterations
        self.use_bias = use_bias
        self.extra_kwargs = dict(kwargs)
        self.progressive_scaler = SimpleNamespace(
            total_iterations=tf.Variable(
                float(progressive_iterations), trainable=False
            ),
            current_iteration=tf.Variable(0.0, trainable=False),
        )
        self.training_calls = []
        self.last_inputs = None
        FakeHierarchicalEncoder.instances.append(self)

    def _parts(self, inputs):
        parts = []
        if self.is_conditioned_on_z:
            parts.append(inputs[Constants.MODULE_INPUTS_CONDITIONED_Z])
        if self.is_conditioned_on_z_hat:
            parts.append(inputs[Constants.MODULE_INPUTS_CONDITIONED_Z_HAT])
        parts.append(inputs[Constants.MODEL_OUTPUTS_Z])
        return parts

    def build(self, input_shape):
        dims = 0
        if self.is_conditioned_on_z:
            dims += input_shape[Constants.MODULE_INPUTS_CONDITIONED_Z][-1]
        if self.is_conditioned_on_z_hat:
            dims += input_shape[Constants.MODULE_INPUTS_CONDITIONED_Z_HAT][-1]
        dims += input_shape[Constants.MODEL_OUTPUTS_Z][-1]
        self.w = self.add_weight(
            shape=(dims, 2 * self.n_latent_dims),
            initializer=tf.keras.initializers.Constant(
                _det_matrix(dims, 2 * self.n_latent_dims, start=0.2, step=0.03)
            ),
            name="w",
        )

    def call(self, inputs, training=None):
        self.training_calls.append(training)
        self.last_inputs = dict(inputs)
        return tf.matmul(tf.concat(self._parts(inputs), axis=-1), self.w)


class FakeDecoder(tf.keras.layers.Layer):
    """matrix @ W -> (batch, n_vars); multiplies by libsize if given."""

    instances = []

    def __init__(
        self, distribution_name, n_vars=5, n_layers=3, name="decoder", **kwargs
    ):
        super().__init__(name=name)
        self.distribution_name = distribution_name
        self.n_vars = n_vars
        self.n_layers = n_layers
        self.training_calls = []
        self.last_inputs = None
        FakeDecoder.instances.append(self)

    def build(self, input_shape):
        in_dims = input_shape[Constants.TENSOR_NAME_X][-1]
        self.w = self.add_weight(
            shape=(in_dims, self.n_vars),
            initializer=tf.keras.initializers.Constant(
                _det_matrix(in_dims, self.n_vars, start=-0.1, step=0.02)
            ),
            name="w",
        )

    def call(self, inputs, training=None):
        self.training_calls.append(training)
        self.last_inputs = dict(inputs)
        out = tf.matmul(inputs[Constants.TENSOR_NAME_X], self.w)
        if Constants.TENSOR_NAME_LIBSIZE in inputs:
            out = out * inputs[Constants.TENSOR_NAME_LIBSIZE]
        return out


# ---------------------------------------------------------------------------
# cavachon.layers fakes
# ---------------------------------------------------------------------------
class FakeMixturePrior(tf.keras.layers.Layer):
    """Returns a trainable (1, n_components, 2 * event_dims + 1) tensor
    regardless of the (1, 1) dummy input."""

    instances = []

    def __init__(self, event_dims, n_components, name="prior", **kwargs):
        super().__init__(name=name)
        self.event_dims = event_dims
        self.n_components = n_components
        self.input_shapes = []
        FakeMixturePrior.instances.append(self)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.n_components, 2 * self.event_dims + 1),
            initializer=tf.keras.initializers.Constant(
                _det_matrix(self.n_components, 2 * self.event_dims + 1, start=0.5)
            ),
            name="w",
        )

    def call(self, inputs):
        self.input_shapes.append(tuple(inputs.shape))
        return tf.expand_dims(self.w, 0) + 0.0 * tf.reduce_sum(inputs)


class FakeSampler(tf.keras.layers.Layer):
    """Returns the loc half of the parameters (+ Z_HAT_SHIFT when built
    deterministic, i.e. as the z_hat sampler)."""

    instances = []

    def __init__(self, name="sampler", stochastic=True, **kwargs):
        super().__init__(name=name)
        self.stochastic = stochastic
        self.training_calls = []
        FakeSampler.instances.append(self)

    def call(self, inputs, training=None):
        self.training_calls.append(training)
        n = inputs.shape[-1] // 2
        shift = 0.0 if self.stochastic else Z_HAT_SHIFT
        return inputs[..., :n] + shift


class FakeScaler:
    """Stand-in for ProgressiveScaler: weight(x) = scale * x."""

    def __init__(self, total_iterations=1, scale=1.0, **kwargs):
        self.total_iterations = total_iterations
        self.scale = float(scale)

    def __call__(self, inputs):
        return tf.constant(self.scale, tf.float32) * tf.cast(inputs, tf.float32)


class FakeToDense:
    """Records the key; densifies inputs[key] if it is a SparseTensor."""

    calls = []

    def __init__(self, key, *args, **kwargs):
        self.key = key

    def __call__(self, inputs):
        FakeToDense.calls.append(self.key)
        out = dict(inputs)
        if isinstance(out[self.key], tf.SparseTensor):
            out[self.key] = tf.sparse.to_dense(out[self.key])
        return out


# ---------------------------------------------------------------------------
# cavachon.losses fakes. Each per-sample loss is
#     weight * (value + coef * mean(y_pred, -1))
# with class-level `value` / `coef`, so the totals are fully controlled.
# ---------------------------------------------------------------------------
class _RecordingLoss(tf.keras.losses.Loss):
    value = 1.0
    coef = 0.0
    instances = []

    def __init__(self, name):
        super().__init__(name=name, reduction="sum_over_batch_size")
        self.calls = []
        type(self).instances.append(self)

    def _weight_tensor(self):
        raise NotImplementedError

    def call(self, y_true, y_pred):
        self.calls.append((y_true, y_pred))
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_2d = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
        per_sample = self.value + self.coef * tf.reduce_mean(y_pred_2d, axis=-1)
        return self._weight_tensor() * per_sample


class FakeStandardKL(_RecordingLoss):
    """Like StandardKLDivergence: weight wrapped in a (fake) ProgressiveScaler."""

    value = 3.0
    instances = []

    def __init__(self, weight=1.0, name="standard_kl_divergence", **kwargs):
        super().__init__(name)
        self.weight = FakeScaler(total_iterations=1, scale=float(weight))

    def _weight_tensor(self):
        return self.weight(tf.ones(()))


class FakeGMMKL(_RecordingLoss):
    """Like GMMKLDivergence: weight wrapped in a (fake) ProgressiveScaler."""

    value = 5.0
    instances = []

    def __init__(self, weight=1.0, name="gmm_kl_divergence", **kwargs):
        super().__init__(name)
        self.weight = FakeScaler(total_iterations=1, scale=float(weight))

    def _weight_tensor(self):
        return self.weight(tf.ones(()))


class FakeNLL(_RecordingLoss):
    """Like NegativeLogDataLikelihood: `weight` is documented as a float and
    snapshotted into a (fake) ProgressiveScaler with scale=float(weight)."""

    value = 7.0
    instances = []

    def __init__(
        self, dist_x_z, weight=1.0, name="negative_log_data_likelihood", **kwargs
    ):
        super().__init__(name)
        self.dist_x_z = dist_x_z
        self.weight_arg = weight
        self.weight = FakeScaler(total_iterations=1, scale=float(weight))

    def _weight_tensor(self):
        return self.weight(tf.ones(()))


class FakeDensity(_RecordingLoss):
    """Like GMMDensityLoss: keeps the given weight object (tf.Variable)."""

    value = 11.0
    instances = []

    def __init__(self, weight=None, name="gmm_density_loss", **kwargs):
        super().__init__(name)
        self.weight = weight

    def _weight_tensor(self):
        if self.weight is None:
            return tf.constant(1.0)
        return tf.cast(self.weight, tf.float32)


FAKE_CLASSES = [
    FakePreprocessor,
    FakeEncoder,
    FakeHierarchicalEncoder,
    FakeDecoder,
    FakeMixturePrior,
    FakeSampler,
    FakeStandardKL,
    FakeGMMKL,
    FakeNLL,
    FakeDensity,
]


@pytest.fixture
def fakes(monkeypatch):
    """Patch every collaborator component.py looks up."""
    for cls in FAKE_CLASSES:
        cls.instances = []
    FakeToDense.calls = []
    for cls, attrs in [
        (FakeStandardKL, (3.0, 0.0)),
        (FakeGMMKL, (5.0, 0.0)),
        (FakeNLL, (7.0, 0.0)),
        (FakeDensity, (11.0, 0.0)),
    ]:
        cls.value, cls.coef = attrs

    patches = {
        "Preprocessor": FakePreprocessor,
        "EncoderLatentParameterizer": FakeEncoder,
        "HierarchicalEncoder": FakeHierarchicalEncoder,
        "DecoderDataParameterizer": FakeDecoder,
        "MixtureMultivariateNormalDiagParameterizerLayer": FakeMixturePrior,
        "MultivariateNormalDiagSampler": FakeSampler,
        "ProgressiveScaler": FakeScaler,
        "ToDense": FakeToDense,
        "StandardKLDivergence": FakeStandardKL,
        "GMMKLDivergence": FakeGMMKL,
        "NegativeLogDataLikelihood": FakeNLL,
        "GMMDensityLoss": FakeDensity,
    }
    for name, fake in patches.items():
        monkeypatch.setattr(component_module, name, fake)
    # TensorUtils (cavachon.utils): remove_nan_gradients passes the
    # gradients through unchanged and records them.
    tensor_utils = MagicMock(name="TensorUtils")
    tensor_utils.remove_nan_gradients.side_effect = lambda gradients: list(gradients)
    monkeypatch.setattr(component_module, "TensorUtils", tensor_utils)
    return SimpleNamespace(
        TensorUtils=tensor_utils,
        Preprocessor=FakePreprocessor,
        Encoder=FakeEncoder,
        HierarchicalEncoder=FakeHierarchicalEncoder,
        Decoder=FakeDecoder,
        Prior=FakeMixturePrior,
        Sampler=FakeSampler,
        Scaler=FakeScaler,
        ToDense=FakeToDense,
        StandardKL=FakeStandardKL,
        GMMKL=FakeGMMKL,
        NLL=FakeNLL,
        Density=FakeDensity,
    )


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def make_component(
    name="comp",
    z_dims=None,
    z_hat_dims=None,
    learn_z_hat_priors=False,
    modalities=None,
    **kwargs,
):
    """Component.make with tiny dims. Requires the `fakes` fixture."""
    modalities = modalities or MODALITIES
    extra = dict(kwargs)
    if learn_z_hat_priors is not False:
        extra["learn_z_hat_priors"] = learn_z_hat_priors
    return component_module.Component.make(
        modality_names=modalities,
        distribution_names={m: DISTRIBUTIONS[m] for m in modalities},
        n_vars={m: N_VARS[m] for m in modalities},
        n_vars_batch_effect={m: N_BATCH_EFFECT[m] for m in modalities},
        n_latent_dims=N_LATENT,
        n_latent_priors=N_PRIORS,
        n_encoder_layers=1,
        n_decoder_layers=1,
        z_conditional_dims=z_dims,
        z_hat_conditional_dims=z_hat_dims,
        conditioned_on_z=["parent_z"] if z_dims is not None else [],
        conditioned_on_z_hat=["parent_z_hat"] if z_hat_dims is not None else [],
        name=name,
        **extra,
    )


def make_batch(modalities=None, z_dims=None, z_hat_dims=None, batch=BATCH, seed=0):
    """Deterministic input batch (non-negative counts-like values)."""
    modalities = modalities or MODALITIES
    rng = np.random.default_rng(seed)
    data = {}
    for m in modalities:
        data[f"{m}_{Constants.TENSOR_NAME_X}"] = tf.constant(
            rng.integers(0, 4, size=(batch, N_VARS[m])).astype(np.float32)
        )
        data[f"{m}_{Constants.TENSOR_NAME_BATCH}"] = tf.constant(
            rng.normal(size=(batch, N_BATCH_EFFECT[m])).astype(np.float32)
        )
    if z_dims is not None:
        data[Constants.MODULE_INPUTS_CONDITIONED_Z] = tf.constant(
            rng.normal(size=(batch, z_dims)).astype(np.float32)
        )
    if z_hat_dims is not None:
        data[Constants.MODULE_INPUTS_CONDITIONED_Z_HAT] = tf.constant(
            rng.normal(size=(batch, z_hat_dims)).astype(np.float32)
        )
    return data


@pytest.fixture
def real_hierarchical_encoder_env(monkeypatch):
    """Lets the real modules.base.HierarchicalEncoder run with its
    cavachon.layers dependencies replaced in its own namespace:
    ProgressiveScaler -> identity with recorded total_iterations,
    MultivariateNormalDiagParameterizerLayer -> identity (returns its
    concatenated input unchanged, so the concat order is observable).
    """

    class IdentityScaler(tf.keras.layers.Layer):
        def __init__(self, total_iterations=5000, scale=1.0, **kwargs):
            super().__init__()
            self.total_iterations = tf.Variable(
                float(total_iterations), trainable=False
            )
            self.current_iteration = tf.Variable(1.0, trainable=False)

        def call(self, inputs, training=None):
            return inputs

    class IdentityParameterizer(tf.keras.layers.Layer):
        def __init__(self, event_dims, name=None, **kwargs):
            super().__init__(name=name)

        def call(self, inputs, training=None):
            return tf.identity(inputs)

    monkeypatch.setattr(he_module, "ProgressiveScaler", IdentityScaler)
    monkeypatch.setattr(
        he_module, "MultivariateNormalDiagParameterizerLayer", IdentityParameterizer
    )
    return SimpleNamespace(Scaler=IdentityScaler, Parameterizer=IdentityParameterizer)


@pytest.fixture
def metrics_shim(monkeypatch):
    """Replaces the Keras-3 `compiled_metrics` backwards-compat shim (a
    third-party property) on Component with a recorder, so train_step can be
    exercised past the line that crashes on Keras 3 (see the xfail in
    test_component_train_step.py). Records (y_true, y_pred) per call."""
    calls = []
    stub = SimpleNamespace(
        update_state=lambda y_true, y_pred: calls.append((y_true, y_pred))
    )
    monkeypatch.setattr(
        component_module.Component, "compiled_metrics", property(lambda self: stub)
    )
    return calls
