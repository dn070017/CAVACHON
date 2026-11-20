"""Shared fakes for the ``cavachon.scheduler`` unit tests.

Everything the scheduler package imports from other cavachon packages
(callbacks, EarlyStoppingConfig, GeneralUtils, the Model and its Components)
and mlflow is
replaced by lightweight recording fakes, so only the scheduler code itself
runs. Loss weights are FakeScaler (ProgressiveScaler-like: pin_to /
activate) or FakeVar (plain Variable: assign).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

import cavachon.scheduler.gmm_prior_initializer as gpi_module
import cavachon.scheduler.sequential_training_scheduler as sts_module


# ---------------------------------------------------------------------------
# Value holders
# ---------------------------------------------------------------------------


class FakeVar:
    """Minimal stand-in for a tf.Variable (assign / float / numpy)."""

    def __init__(self, value=0.0):
        self.value = np.asarray(value, dtype=np.float32)

    def assign(self, value):
        self.value = np.array(value, dtype=np.float32)

    def numpy(self):
        return self.value

    def __float__(self):
        return float(self.value)


class FakeScaler:
    """Stand-in for ProgressiveScaler; records pin_to / activate calls.

    ``state`` is ``("pin", value)`` after ``pin_to`` or ``("active", n)``
    after ``activate``.
    """

    def __init__(self, scale=1.0):
        self.scale = FakeVar(scale)
        self.current_iteration = FakeVar(1.0)
        self.total_iterations = FakeVar(1.0)
        self.calls = []
        self.state = ("pin", float(scale))

    def pin_to(self, value):
        self.calls.append(("pin", float(value)))
        self.state = ("pin", float(value))
        self.total_iterations.assign(1.0)
        self.current_iteration.assign(float(value) / max(float(self.scale), 1e-7))

    def activate(self, total_iterations):
        self.calls.append(("active", float(total_iterations)))
        self.state = ("active", float(total_iterations))
        self.total_iterations.assign(float(total_iterations))
        self.current_iteration.assign(0.0)


# ---------------------------------------------------------------------------
# Model / component fakes
# ---------------------------------------------------------------------------


class FakeLayer:
    def __init__(self, name):
        self.name = name
        self.trainable = True
        self.var = object()  # identity used as a "trainable variable"
        self.submodules = []
        self.layers = []


class FakePrior(FakeLayer):
    def __init__(self, name, n_components=2, event_dims=2):
        super().__init__(name)
        self.n_components = n_components
        self.loc_bias = [
            FakeVar(np.zeros((1, event_dims))) for _ in range(n_components)
        ]
        self.scale_diag_bias = [
            FakeVar(np.zeros((1, event_dims))) for _ in range(n_components)
        ]
        self.logits_bias = FakeVar(np.zeros((1, n_components)))


class FakeComponent:
    def __init__(self, name, learn_z_hat_priors=False, n_clusters=2, event_dims=2):
        self.name = name
        self.trainable = True
        self.learn_z_hat_priors = learn_z_hat_priors
        self.preprocessor = FakeLayer(f"{name}_preprocessor")
        self.encoder = FakeLayer(f"{name}_encoder")
        self.decoder = FakeLayer(f"{name}_decoder")
        self.inner = FakeLayer(f"{name}_inner")
        self.decoder.layers = [self.inner]  # nested: reached through .layers
        self.z_prior_parameterizer = FakePrior(
            f"{name}_z_prior", n_clusters, event_dims
        )
        self.z_hat_prior_parameterizer = (
            FakePrior(f"{name}_z_hat_prior", n_clusters, event_dims)
            if learn_z_hat_priors
            else None
        )
        self.hierarchical_encoder = SimpleNamespace(
            progressive_scaler=SimpleNamespace(
                current_iteration=FakeVar(1.0), total_iterations=FakeVar(1.0)
            ),
            initialize_scale_diag=MagicMock(name=f"{name}.initialize_scale_diag"),
        )
        self.z_sampler = SimpleNamespace(stochastic=FakeVar(1.0))
        self.z_hat_sampler = SimpleNamespace(stochastic=FakeVar(0.0))
        self.submodules = [
            self.preprocessor,
            self.encoder,
            self.decoder,
            self.z_prior_parameterizer,
        ] + ([self.z_hat_prior_parameterizer] if learn_z_hat_priors else [])
        self.progressive_calls = []

    def set_progressive_scaler_iteration(self, current_iteration=1, total_iterations=1):
        self.progressive_calls.append(
            (float(current_iteration), float(total_iterations))
        )
        ps = self.hierarchical_encoder.progressive_scaler
        ps.current_iteration.assign(float(current_iteration))
        ps.total_iterations.assign(float(total_iterations))

    @property
    def progressive(self):
        ps = self.hierarchical_encoder.progressive_scaler
        return (float(ps.current_iteration), float(ps.total_iterations))

    def all_layers(self):
        return self.submodules + [self.inner]


def make_config(
    name,
    z=(),
    z_hat=(),
    n_vars=None,
    n_parent_annealing_epochs=0,
    n_kl_annealing_epochs=0,
    enable_kmeans_init=False,
    max_regular_training_epochs=5,
    kl_annealing_ratio=None,
    n_z_hat_annealing_epochs=0,
    n_z_hat_training_epochs=0,
    z_hat_kl_weight=1.0,
):
    return SimpleNamespace(
        name=name,
        conditioned_on_z=list(z),
        conditioned_on_z_hat=list(z_hat),
        n_vars=n_vars if n_vars is not None else {"rna": 10},
        n_parent_annealing_epochs=n_parent_annealing_epochs,
        n_kl_annealing_epochs=n_kl_annealing_epochs,
        enable_kmeans_init=enable_kmeans_init,
        max_regular_training_epochs=max_regular_training_epochs,
        kl_annealing_ratio=kl_annealing_ratio,
        n_z_hat_annealing_epochs=n_z_hat_annealing_epochs,
        n_z_hat_training_epochs=n_z_hat_training_epochs,
        z_hat_kl_weight=z_hat_kl_weight,
    )


class FakeModel:
    """Records compile / fit / predict calls and snapshots the training
    state (trainable flags, loss weights, progressive scalers) at the
    start of each ``fit``."""

    def __init__(self, configs, learn_z_hat=(), n_clusters=2, event_dims=2):
        self.name = "fake_model"
        self.component_configs = configs
        self.components = {
            c.name: FakeComponent(
                c.name,
                c.name in learn_z_hat,
                n_clusters=n_clusters,
                event_dims=event_dims,
            )
            for c in configs
        }
        self._standard_kl_weights = {c.name: FakeScaler(1.0) for c in configs}
        self._gmm_kl_weights = {c.name: FakeScaler(1.0) for c in configs}
        self._data_loss_weights = {
            c.name: {mod: FakeScaler(1.0) for mod in c.n_vars} for c in configs
        }
        self._z_hat_density_weights = {n: FakeVar(1.0) for n in learn_z_hat}
        self._z_hat_gmm_kl_weights = {
            c.name: FakeScaler(c.z_hat_kl_weight)
            for c in configs
            if c.name in learn_z_hat
        }
        self.optimizer = None
        self.compile_calls = []
        self.fit_calls = []
        self.trainable_history = []
        self.predict_outputs = {}
        # list of ints: number of epochs actually run for successive fits
        # (simulates early stopping); None entries mean "all epochs"
        self.epochs_run = []
        self.fit_side_effect = None

    # model.trainable is toggled by the k-means initialisation
    @property
    def trainable(self):
        return self.trainable_history[-1] if self.trainable_history else True

    @trainable.setter
    def trainable(self, value):
        self.trainable_history.append(value)

    @property
    def trainable_variables(self):
        out = []
        for comp in self.components.values():
            for layer in comp.all_layers():
                if layer.trainable:
                    out.append(layer.var)
        return out

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        self.optimizer = kwargs.get("optimizer")

    def snapshot(self):
        snap = {}
        for name, comp in self.components.items():
            snap[name] = dict(
                trainable=comp.trainable,
                layers={layer.name: layer.trainable for layer in comp.all_layers()},
                progressive=comp.progressive,
                standard_kl=self._standard_kl_weights[name].state,
                gmm_kl=self._gmm_kl_weights[name].state,
                data={m: s.state for m, s in self._data_loss_weights[name].items()},
                z_hat_gmm_kl=(
                    self._z_hat_gmm_kl_weights[name].state
                    if name in self._z_hat_gmm_kl_weights
                    else None
                ),
                stochastic=float(comp.z_hat_sampler.stochastic),
                z_stochastic=float(comp.z_sampler.stochastic),
            )
        return snap

    def fit(self, x, epochs=1, callbacks=None, verbose=0, **kwargs):
        if self.fit_side_effect is not None:
            raise self.fit_side_effect
        self.fit_calls.append(
            SimpleNamespace(
                x=x,
                epochs=epochs,
                callbacks=list(callbacks or []),
                verbose=verbose,
                kwargs=kwargs,
                state=self.snapshot(),
                optimizer=self.optimizer,
            )
        )
        n_run = epochs
        if self.epochs_run:
            override = self.epochs_run.pop(0)
            if override is not None:
                n_run = override
        return SimpleNamespace(epoch=list(range(n_run)))

    def predict(self, mdata, batch_size=None, verbose=0):
        self.predict_args = (mdata, batch_size, verbose)
        return self.predict_outputs


# ---------------------------------------------------------------------------
# GeneralUtils.order_components fake
# ---------------------------------------------------------------------------


def fake_order_components(configs):
    """Reference topological sort with GeneralUtils.order_components's
    contract: parents before children, config order kept otherwise,
    ValueError for an unknown parent, AttributeError for a cycle."""
    configs = list(configs)
    names = {c.name for c in configs}
    ordered, done = [], set()
    remaining = list(configs)
    while remaining:
        progressed = False
        for config in list(remaining):
            parents = list(config.conditioned_on_z) + list(config.conditioned_on_z_hat)
            unknown = [p for p in parents if p not in names]
            if unknown:
                raise ValueError(f"unknown component {unknown[0]!r}")
            if all(p in done for p in parents):
                ordered.append(config)
                done.add(config.name)
                remaining.remove(config)
                progressed = True
        if not progressed:
            raise AttributeError("cycle")
    return ordered


# ---------------------------------------------------------------------------
# Callback fakes
# ---------------------------------------------------------------------------


def _make_recording_callback(cls_name):
    class _Recorder:
        instances = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            type(self).instances.append(self)

        def __repr__(self):
            return f"<{cls_name} {self.kwargs}>"

    _Recorder.__name__ = cls_name
    _Recorder.instances = []
    return _Recorder


class UserCallback:
    """A user-supplied callback passed via fit(callbacks=[...])."""

    def __init__(self, tag):
        self.tag = tag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fakes(monkeypatch):
    """Patch every cross-package dependency of the scheduler module."""
    ns = SimpleNamespace()
    for cb in (
        "AnnealingCallback",
        "EarlyStoppingCallback",
        "VerboseCallback",
    ):
        rec = _make_recording_callback(cb)
        setattr(ns, cb, rec)
        monkeypatch.setattr(sts_module, cb, rec)
    ns.VerboseCallback._COLORS = ["<c0>", "<c1>", "<c2>"]
    ns.VerboseCallback._BOLD = "<b>"
    ns.VerboseCallback._RESET = "<r>"
    # GMMPriorInitializer reads the colors of VerboseCallback
    monkeypatch.setattr(gpi_module, "VerboseCallback", ns.VerboseCallback)
    # EarlyStoppingConfig lives in cavachon.config: a fake with the same
    # documented defaults stands in for it.
    ns.EarlyStoppingConfig = MagicMock(
        name="EarlyStoppingConfig",
        side_effect=lambda: SimpleNamespace(
            monitor="loss", mode="min", patience=25, min_delta=5.0
        ),
    )
    monkeypatch.setattr(sts_module, "EarlyStoppingConfig", ns.EarlyStoppingConfig)
    ns.GeneralUtils = MagicMock(name="GeneralUtils")
    ns.GeneralUtils.order_components.side_effect = fake_order_components
    monkeypatch.setattr(sts_module, "GeneralUtils", ns.GeneralUtils)
    ns.mlflow = MagicMock(name="mlflow")
    ns.mlflow.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="exp-1"
    )
    monkeypatch.setattr(sts_module, "mlflow", ns.mlflow)
    return ns


@pytest.fixture
def dataset():
    """A 3-batch dataset (len() == 3)."""
    return tf.data.Dataset.range(3).batch(1)


def make_scheduler(configs, learn_z_hat=(), **kwargs):
    model = FakeModel(configs, learn_z_hat=learn_z_hat)
    scheduler = sts_module.SequentialTrainingScheduler(model, mdata="MDATA", **kwargs)
    return scheduler, model


def callbacks_of(call, cls):
    return [cb for cb in call.callbacks if isinstance(cb, cls)]
