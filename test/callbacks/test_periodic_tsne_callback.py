"""Unit tests for cavachon.callbacks.periodic_tsne_callback.

Mocked (patched where periodic_tsne_callback looks them up):
``os.makedirs``, ``np.save``, ``tqdm``, ``DataLoader``,
``MultivariateNormalDiagDistribution`` and
``MixtureMultivariateNormalDiagDistribution``. The model and MuData are
MagicMocks. No file is created.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

import cavachon.callbacks.periodic_tsne_callback as ptc
from cavachon.callbacks.periodic_tsne_callback import PeriodicTSNECallback

N_COMPONENTS, LATENT = 3, 2


@pytest.fixture
def patched(monkeypatch):
    mocks = {
        "makedirs": MagicMock(name="makedirs"),
        "save": MagicMock(name="np.save"),
        "DataLoader": MagicMock(name="DataLoader"),
        "MVN": MagicMock(name="MultivariateNormalDiagDistribution"),
        "Mixture": MagicMock(name="MixtureMultivariateNormalDiagDistribution"),
    }
    monkeypatch.setattr(ptc.os, "makedirs", mocks["makedirs"])
    monkeypatch.setattr(ptc.np, "save", mocks["save"])
    monkeypatch.setattr(ptc, "tqdm", lambda it, **kw: it)
    monkeypatch.setattr(ptc, "DataLoader", mocks["DataLoader"])
    monkeypatch.setattr(ptc, "MultivariateNormalDiagDistribution", mocks["MVN"])
    monkeypatch.setattr(
        ptc, "MixtureMultivariateNormalDiagDistribution", mocks["Mixture"]
    )
    return mocks


def _wire_model(mocks, n_batches=2, batch=4):
    """A fake model + distributions so the full on_epoch_end body could run."""
    model = MagicMock(name="model")
    model.trainable = True
    model.predict.return_value = {"c_z": np.zeros((n_batches * batch, LATENT))}
    params = tf.zeros((1, N_COMPONENTS, 1 + 2 * LATENT))
    model.components = {"c": MagicMock(z_prior_parameterizer=lambda x: params)}
    model.side_effect = lambda b, training=False: {"c_z": tf.zeros((batch, LATENT))}
    mocks["DataLoader"].return_value = [object()] * n_batches
    mocks["MVN"].from_parameterizer_output.return_value.log_prob.side_effect = (
        lambda z: tf.zeros((batch, N_COMPONENTS))
    )
    mocks["Mixture"].from_parameterizer_output.return_value.log_prob.side_effect = (
        lambda z: tf.zeros((batch,))
    )
    return model


def test_init_stores_arguments_and_creates_outdir(patched):
    mdata = MagicMock(name="mdata")
    cb = PeriodicTSNECallback(
        mdata=mdata,
        component="c",
        outdir="/nonexistent/snapshots",
        batch_size=64,
        batch_effect_colnames={"rna": ["batch"]},
        distribution_names={"rna": "IndependentZeroInflatedNegativeBinomial"},
    )
    assert cb.mdata is mdata
    assert cb.component == "c"
    assert cb.batch_size == 64
    assert cb.output_dir == "/nonexistent/snapshots"
    assert cb.batch_effect_colnames == {"rna": ["batch"]}
    assert cb.distribution_names == {"rna": "IndependentZeroInflatedNegativeBinomial"}
    patched["makedirs"].assert_called_once_with("/nonexistent/snapshots", exist_ok=True)


def test_on_epoch_end_is_currently_a_noop(patched):
    cb = PeriodicTSNECallback(MagicMock(), "c", "/nonexistent", batch_size=8)
    model = _wire_model(patched)
    cb.set_model(model)
    for epoch in range(50):
        cb.on_epoch_end(epoch, {"loss": 1.0})
    model.predict.assert_not_called()
    model.assert_not_called()
    assert model.trainable is True
    patched["DataLoader"].assert_not_called()
    patched["save"].assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: save_epochs is hard-coded to an empty set, so the callback "
    "never saves anything despite being added whenever output_dir is set",
)
def test_saves_snapshots_periodically(patched):
    cb = PeriodicTSNECallback(MagicMock(), "c", "/nonexistent", batch_size=8)
    cb.set_model(_wire_model(patched))
    for epoch in range(50):
        cb.on_epoch_end(epoch, {"loss": 1.0})
    assert patched["save"].call_count > 0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: `batch_size=int` (default is the builtin type) instead of "
    "the annotation `batch_size: int`",
)
def test_batch_size_is_required_or_an_int(patched):
    try:
        cb = PeriodicTSNECallback(MagicMock(), "c", "/nonexistent")
    except TypeError:
        return
    assert isinstance(cb.batch_size, int)
