"""Unit tests for cavachon.dataloader.modifiers.

``cavachon.layers.modifiers.ToDense`` / ``Binarize`` are replaced by fakes
(see ``fake_layers`` in conftest.py); these tests check that each data
modifier constructs the right layers with the right key and chains them.
"""

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.dataloader.modifiers import (
    IndependentBernoulliDataModifier,
    IndependentZeroInflatedNegativeBinomialDataModifier,
    MultivariateNormalDiagDataModifier,
    StudenttDataModifier,
)
from test.dataloader.conftest import (
    MODIFIER_CLASSES,
    MODIFIER_MODULES,
    FakeBinarize,
    FakeToDense,
)

ALL = list(MODIFIER_CLASSES.values())
TODENSE_ONLY = [
    IndependentZeroInflatedNegativeBinomialDataModifier,
    MultivariateNormalDiagDataModifier,
    StudenttDataModifier,
]
X = np.array([[0.0, 0.5, 1.0, 3.0], [2.0, 0.0, 0.99, 7.25]], dtype=np.float32)


def sparse_inputs(mod="m"):
    return {
        f"{mod}_matrix": tf.sparse.from_dense(tf.constant(X)),
        f"{mod}_batch_effect": tf.ones((2, 1)),
    }


class Recorder:
    """Plain callable that logs (name, inputs) and returns a tagged copy."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __call__(self, inputs):
        self.log.append((self.name, inputs))
        out = dict(inputs)
        out["trace"] = out.get("trace", ()) + (self.name,)
        return out


# --------------------------------------------------------------------------
# wiring: which layers are built, with which key
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
def test_modality_key(cls):
    assert cls("rna").modality_key == "rna_matrix"


def test_bernoulli_builds_todense_then_binarize(fake_layers):
    mod = IndependentBernoulliDataModifier("atac")
    assert mod.modality_name == "atac"
    fake_layers.ToDense.assert_called_once_with("atac_matrix")
    fake_layers.Binarize.assert_called_once_with("atac_matrix")
    assert [type(m) for m in mod.modifiers] == [FakeToDense, FakeBinarize]
    assert FakeBinarize.instances[0].threshold == 1.0  # Binarize's own default is used


@pytest.mark.parametrize("cls", TODENSE_ONLY)
def test_todense_only_modifiers_build_single_todense(cls, fake_layers):
    mod = cls("rna")
    fake_layers.ToDense.assert_called_once_with("rna_matrix")
    fake_layers.Binarize.assert_not_called()
    assert [type(m) for m in mod.modifiers] == [FakeToDense]


@pytest.mark.parametrize("name", list(MODIFIER_MODULES))
def test_layers_looked_up_in_modifier_module(name, monkeypatch):
    """Patching the module-level name must be what the modifier uses."""
    sentinel_factory = mock.MagicMock(side_effect=FakeToDense)
    monkeypatch.setattr(f"{MODIFIER_MODULES[name]}.ToDense", sentinel_factory)
    MODIFIER_CLASSES[name]("m")
    sentinel_factory.assert_called_once_with("m_matrix")


# --------------------------------------------------------------------------
# call(): chaining semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
def test_call_chains_modifiers_in_order(cls):
    log = []
    mod = cls("m")
    names = [f"layer{i}" for i in range(len(mod.modifiers))]
    mod.modifiers = [Recorder(n, log) for n in names]
    inputs = sparse_inputs()
    out = mod(inputs)
    assert [n for n, _ in log] == names
    # first layer gets the original mapping, next one gets the previous output
    assert log[0][1]["m_matrix"] is inputs["m_matrix"]
    assert "trace" not in log[0][1]
    for i in range(1, len(log)):
        assert log[i][1]["trace"] == tuple(names[:i])
    assert out["trace"] == tuple(names)


@pytest.mark.parametrize("cls", ALL)
def test_each_layer_called_once_per_call(cls):
    mod = cls("m")
    mod(sparse_inputs())
    assert all(layer.calls == 1 for layer in mod.modifiers)


def test_bernoulli_output_with_fakes():
    out = IndependentBernoulliDataModifier("m")(sparse_inputs())
    assert not isinstance(out["m_matrix"], tf.SparseTensor)
    np.testing.assert_array_equal(
        out["m_matrix"].numpy(), (X >= 1.0).astype(np.float32)
    )
    np.testing.assert_array_equal(out["m_batch_effect"].numpy(), np.ones((2, 1)))


@pytest.mark.parametrize("cls", TODENSE_ONLY)
def test_todense_modifiers_output_with_fakes(cls):
    out = cls("m")(sparse_inputs())
    assert not isinstance(out["m_matrix"], tf.SparseTensor)
    np.testing.assert_array_equal(out["m_matrix"].numpy(), X)


@pytest.mark.parametrize("cls", ALL)
def test_modifier_ignores_other_modalities(cls):
    out = cls("m")({**sparse_inputs("m"), **sparse_inputs("other")})
    assert isinstance(out["other_matrix"], tf.SparseTensor)


@pytest.mark.parametrize("cls", ALL)
def test_modifier_missing_key_raises(cls):
    with pytest.raises(KeyError):
        cls("absent")(sparse_inputs("m"))


@pytest.mark.parametrize("cls", ALL)
def test_modifier_traceable_inside_dataset_map(cls):
    ds = tf.data.Dataset.from_tensor_slices(sparse_inputs()).map(cls("m"))
    rows = np.stack([e["m_matrix"].numpy() for e in ds])
    expected = (
        (X >= 1.0).astype(np.float32) if cls is IndependentBernoulliDataModifier else X
    )
    np.testing.assert_array_equal(rows, expected)


# regression: StudenttDataModifier did not set `modality_name`
def test_studentt_modifier_exposes_modality_name():
    assert getattr(StudenttDataModifier("m"), "modality_name", None) == "m"
