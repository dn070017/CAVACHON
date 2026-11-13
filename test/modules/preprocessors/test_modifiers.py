"""Unit tests for cavachon.modules.preprocessors.modifiers.*.

Mocked: ToDense, LogTransform, NormalizeLibrarySize, Binarize
(cavachon.layers.modifiers) -> arithmetic fakes from conftest.
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.modules.preprocessors.modifiers import (
    IndependentBernoulliModifier,
    IndependentZeroInflatedNegativeBinomialModifier,
    MultivariateNormalDiagModifier,
    StudenttModifier,
)
from test.modules.preprocessors.conftest import (
    FakeBinarize,
    FakeLogTransform,
    FakeNormalizeLibrarySize,
    FakeToDense,
)

X = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
OTHER = np.array([[7.0], [8.0]], dtype=np.float32)


def _assert_passes(log, expected):
    """Keras may run call() more than once (build trace + eager); every
    pass must apply the modifiers in exactly the expected order."""
    n = len(expected)
    assert len(log) >= n and len(log) % n == 0
    for i in range(0, len(log), n):
        assert log[i : i + n] == expected


def _inputs():
    return {"rna_matrix": tf.constant(X), "atac_matrix": tf.constant(OTHER)}


@pytest.mark.parametrize(
    "cls, expected_types",
    [
        (IndependentBernoulliModifier, [FakeToDense, FakeBinarize]),
        (
            IndependentZeroInflatedNegativeBinomialModifier,
            [FakeToDense, FakeLogTransform, FakeNormalizeLibrarySize],
        ),
        (MultivariateNormalDiagModifier, [FakeToDense]),
        (StudenttModifier, [FakeToDense]),
    ],
)
def test_modifier_list_and_keys(fake_modifier_layers, cls, expected_types):
    m = cls(modality_name="rna")
    assert m.modality_name == "rna"
    assert m.modality_key == "rna_matrix"
    assert [type(x) for x in m.modifiers] == expected_types
    assert all(x.key == "rna_matrix" for x in m.modifiers)


def test_bernoulli_order_todense_then_binarize(fake_modifier_layers):
    out = IndependentBernoulliModifier("rna")(_inputs())
    np.testing.assert_allclose(out["rna_matrix"].numpy(), (X + 1) * 10)
    np.testing.assert_allclose(out["atac_matrix"].numpy(), OTHER)
    _assert_passes(
        fake_modifier_layers, [("FakeToDense", "rna_matrix"), ("FakeBinarize", "rna_matrix")]
    )


def test_zinb_order_todense_log_normalize(fake_modifier_layers):
    out = IndependentZeroInflatedNegativeBinomialModifier("rna")(_inputs())
    np.testing.assert_allclose(out["rna_matrix"].numpy(), (X + 1) * 2 - 3)
    # libsize is computed on the data entering NormalizeLibrarySize
    np.testing.assert_allclose(
        out["rna_matrix_libsize"].numpy(), ((X + 1) * 2).sum(-1, keepdims=True)
    )
    np.testing.assert_allclose(out["atac_matrix"].numpy(), OTHER)
    _assert_passes(
        fake_modifier_layers,
        [
            ("FakeToDense", "rna_matrix"),
            ("FakeLogTransform", "rna_matrix"),
            ("FakeNormalizeLibrarySize", "rna_matrix"),
        ],
    )


@pytest.mark.parametrize("cls", [MultivariateNormalDiagModifier, StudenttModifier])
def test_dense_only_modifiers(fake_modifier_layers, cls):
    out = cls("rna")(_inputs())
    np.testing.assert_allclose(out["rna_matrix"].numpy(), X + 1)
    assert set(out) == {"rna_matrix", "atac_matrix"}
    _assert_passes(fake_modifier_layers, [("FakeToDense", "rna_matrix")])


def test_modifier_only_touches_own_modality(fake_modifier_layers):
    out = MultivariateNormalDiagModifier("atac")(_inputs())
    np.testing.assert_allclose(out["atac_matrix"].numpy(), OTHER + 1)
    np.testing.assert_allclose(out["rna_matrix"].numpy(), X)
