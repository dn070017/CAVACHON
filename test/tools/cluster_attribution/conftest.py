"""Shared fakes for ``cavachon.tools.cluster_analysis`` and
``cavachon.tools.attribution_analysis`` unit tests.

Everything these modules import from other cavachon packages is replaced
where it is looked up:

cluster_analysis
* ``DataLoader``                                 -> ``MagicMock`` (iterable of batches)
* ``MultivariateNormalDiagDistribution``         -> ``FakeMVNDiag`` (real tfp
                                                    ``MultivariateNormalDiag``, records kwargs)
* ``MixtureMultivariateNormalDiagDistribution``  -> ``FakeMixture`` (real tfp
                                                    ``MixtureSameFamily`` built from the
                                                    ``[logit, loc, scale]`` layout)

attribution_analysis
* ``DataLoader``                    -> ``MagicMock`` whose ``dataset`` is a real
                                       (unbatched) ``tf.data.Dataset``
* ``MultivariateNormalDiagSampler`` -> ``FakeSampler`` (returns the loc)
* ``Model``                         -> ``MagicMock`` (``prepare_component_inputs``)
* ``Component``                     -> ``MagicMock`` (``prepare_*_inputs``)

``cavachon.environment.Constants`` stays real.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import cavachon.tools.attribution_analysis as attribution_module
import cavachon.tools.cluster_analysis as cluster_module
from cavachon.environment.constants import Constants

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_mdata(n_obs=4, n_vars=3, modality="rna", obs=None):
    x = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(obs or {}, index=[f"cell{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene{j}" for j in range(n_vars)]),
    )
    return mu.MuData({modality: adata})


# ---------------------------------------------------------------------------
# cluster_analysis fakes
# ---------------------------------------------------------------------------


class FakeMVNDiag(tfp.distributions.MultivariateNormalDiag):
    calls: list = []

    def __init__(self, loc, scale_diag):
        FakeMVNDiag.calls.append({"loc": loc, "scale_diag": scale_diag})
        super().__init__(loc=loc, scale_diag=scale_diag)


class FakeMixture:
    calls: list = []

    @classmethod
    def from_parameterizer_output(cls, params):
        params = tf.convert_to_tensor(params)
        FakeMixture.calls.append(params)
        d = (params.shape[-1] - 1) // 2
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(logits=params[..., 0]),
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=params[..., 1 : d + 1], scale_diag=params[..., d + 1 :]
            ),
        )


@pytest.fixture
def cluster_fakes(monkeypatch):
    FakeMVNDiag.calls = []
    FakeMixture.calls = []
    dataloader = MagicMock(name="DataLoader")
    monkeypatch.setattr(cluster_module, "DataLoader", dataloader)
    monkeypatch.setattr(
        cluster_module, "MultivariateNormalDiagDistribution", FakeMVNDiag
    )
    monkeypatch.setattr(
        cluster_module, "MixtureMultivariateNormalDiagDistribution", FakeMixture
    )
    return SimpleNamespace(
        DataLoader=dataloader, mvn_calls=FakeMVNDiag.calls, mix_calls=FakeMixture.calls
    )


def make_prior(logits, mu, scale):
    """Fake prior parameterizer with the ``parameters`` property layout
    ``[logit, loc..., scale...]`` per component."""
    params = np.concatenate(
        [np.asarray(logits, np.float32)[:, None], np.asarray(mu), np.asarray(scale)],
        axis=1,
    ).astype(np.float32)
    return SimpleNamespace(parameters=tf.constant(params))


class FakeLatentModel:
    """Callable model: outputs[key] = batch['rna_matrix'] (the 'latent')."""

    def __init__(self, components, keys):
        self.components = components
        self.keys = keys
        self.calls = []

    def __call__(self, batch, training=False):
        self.calls.append(training)
        return {k: tf.convert_to_tensor(batch["rna_matrix"]) for k in self.keys}


# ---------------------------------------------------------------------------
# attribution_analysis fakes
# ---------------------------------------------------------------------------


class FakeSampler:
    """MultivariateNormalDiagSampler stand-in: returns the loc half."""

    def __call__(self, z_parameters, training=False):
        d = z_parameters.shape[-1] // 2
        return z_parameters[..., :d]


class FakePreprocessor:
    matrix_key = "matrix"

    def __call__(self, inputs, training=False):
        return {"matrix": inputs["rna_matrix"]}


class FakeEncoder:
    """z_parameters = [x @ W, ones] so the sampler returns x @ W."""

    def __init__(self, w):
        self.w = tf.constant(w, tf.float32)

    def __call__(self, x, training=False):
        loc = tf.matmul(tf.cast(x, tf.float32), self.w)
        return tf.concat([loc, tf.ones_like(loc)], axis=-1)


class FakeDecoder:
    """attribution target = fn(z_hat)."""

    def __init__(self, fn):
        self.fn = fn
        self.inputs = []

    def compute_attribution_target(self, inputs):
        self.inputs.append(inputs)
        return self.fn(inputs["z_hat"])


class FakeComponentNetwork:
    """hierarchical z_hat = z + sum of z_hat of the parents."""

    def __init__(self, enc_w, decoder_fn=None, parents=()):
        self.preprocessor = FakePreprocessor()
        self.encoder = FakeEncoder(enc_w)
        self.decoders = {"rna": FakeDecoder(decoder_fn)} if decoder_fn else {}
        self.parents = list(parents)
        self.hier_calls = []

    def hierarchical_encode(self, inputs, z, training=False):
        self.hier_calls.append(z)
        z_hat = z
        for p in self.parents:
            z_hat = z_hat + inputs["z_hat_conditional"][p]
        return {Constants.MODEL_OUTPUTS_Z_HAT: z_hat}


class FakeAttrModel:
    def __init__(self, configs, components):
        self.component_configs = configs
        self.components = components

    def __call__(self, batch, training=False):
        x = tf.cast(batch["rna_matrix"], tf.float32)
        return {
            f"{c.name}_{Constants.MODEL_OUTPUTS_Z}": tf.matmul(
                x, self.components[c.name].encoder.w
            )
            for c in self.component_configs
        }


def _prepare_component_inputs(
    batch,
    component_config,
    target_component,
    components,
    z_conditional,
    z_hat_conditional,
):
    return {
        "rna_matrix": batch["rna_matrix"],
        "z_conditional": dict(z_conditional),
        "z_hat_conditional": dict(z_hat_conditional),
    }


@pytest.fixture
def attr_fakes(monkeypatch):
    dataloader = MagicMock(name="DataLoader")
    model_cls = MagicMock(name="Model")
    model_cls.prepare_component_inputs.side_effect = _prepare_component_inputs
    component_cls = MagicMock(name="Component")
    component_cls.prepare_preprocessor_inputs.side_effect = lambda inputs, names: inputs
    component_cls.prepare_decoder_inputs.side_effect = (
        lambda batch, modality_name, z_hat, preprocessor_outputs: {"z_hat": z_hat}
    )
    monkeypatch.setattr(attribution_module, "DataLoader", dataloader)
    monkeypatch.setattr(attribution_module, "Model", model_cls)
    monkeypatch.setattr(attribution_module, "Component", component_cls)
    monkeypatch.setattr(
        attribution_module, "MultivariateNormalDiagSampler", FakeSampler
    )
    return SimpleNamespace(
        DataLoader=dataloader, Model=model_cls, Component=component_cls
    )


def set_dataset(dataloader_mock, x):
    instance = MagicMock(name="dataloader_instance")
    instance.dataset = tf.data.Dataset.from_tensor_slices(
        {"rna_matrix": tf.constant(x, tf.float32)}
    )
    dataloader_mock.return_value = instance
    return instance
