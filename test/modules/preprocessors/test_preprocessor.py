"""Unit tests for cavachon.modules.preprocessors.preprocessor.Preprocessor.

Mocked: ReflectionHandler.get_class_by_name (returns the real in-package
modifier class) and the cavachon.layers.modifiers layers used by those
modifiers (arithmetic fakes from conftest). The per-modality Dense layers
are real; their kernels are set to known values for value checks.
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.modules.preprocessors import Preprocessor

X = Constants.TENSOR_NAME_X


def _set_dense(model, layer_name, value):
    seq = model.get_layer(layer_name)
    dense = seq.layers[0]
    dense.kernel.assign(np.full(dense.kernel.shape, value, dtype=np.float32))
    dense.bias.assign(np.zeros(dense.bias.shape, dtype=np.float32))
    return dense


RNA = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
ATAC = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

DISTS = {"rna": "IndependentZeroInflatedNegativeBinomial", "atac": "IndependentBernoulli"}
NVARS = {"rna": 4, "atac": 3}


@pytest.fixture
def two_modality(modifier_lookup):
    model = Preprocessor.make(["rna", "atac"], DISTS, NVARS, n_dims=2, name="pre")
    return model, modifier_lookup


class TestMake:
    def test_reflection_lookup_per_modality(self, two_modality):
        _, lookup = two_modality
        assert [c.args for c in lookup.call_args_list] == [
            ("IndependentZeroInflatedNegativeBinomial", "modules/preprocessors/modifiers", "Modifier"),
            ("IndependentBernoulli", "modules/preprocessors/modifiers", "Modifier"),
        ]

    def test_inputs_outputs_and_attributes(self, two_modality):
        model, _ = two_modality
        assert isinstance(model, Preprocessor)
        assert model.name == "pre"
        assert model.matrix_key == X
        assert set(model.input) == {"rna_matrix", "atac_matrix"}
        assert tuple(model.input["rna_matrix"].shape) == (None, 4)
        assert tuple(model.input["atac_matrix"].shape) == (None, 3)
        # libsize only for the modality whose modifier produces it (ZINB)
        assert set(model.output) == {X, "rna_matrix_libsize"}
        assert tuple(model.output[X].shape) == (None, 4)  # 2 modalities * n_dims

    def test_dense_layers(self, two_modality):
        model, _ = two_modality
        for m, n in NVARS.items():
            seq = model.get_layer(f"pre_{m}")
            assert len(seq.layers) == 1
            dense = seq.layers[0]
            assert dense.name == f"pre_{m}_dense" and dense.units == 2
            assert dense.kernel.shape == (n, 2)

    def test_values_and_concat_order(self, two_modality):
        model, _ = two_modality
        _set_dense(model, "pre_rna", 1.0)
        _set_dense(model, "pre_atac", 0.5)
        out = model({"rna_matrix": tf.constant(RNA), "atac_matrix": tf.constant(ATAC)})
        rna_mod = (RNA + 1) * 2 - 3          # ToDense -> Log -> Normalize (fakes)
        atac_mod = (ATAC + 1) * 10           # ToDense -> Binarize (fakes)
        expected = np.concatenate(
            [np.repeat(rna_mod.sum(-1, keepdims=True), 2, -1),
             np.repeat(0.5 * atac_mod.sum(-1, keepdims=True), 2, -1)],
            axis=-1,
        )
        np.testing.assert_allclose(out[X].numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose(
            out["rna_matrix_libsize"].numpy(), ((RNA + 1) * 2).sum(-1, keepdims=True)
        )

    def test_concat_follows_modality_names_order(self, modifier_lookup):
        model = Preprocessor.make(["atac", "rna"], DISTS, NVARS, n_dims=1, name="pre")
        _set_dense(model, "pre_rna", 1.0)
        _set_dense(model, "pre_atac", 1.0)
        out = model({"rna_matrix": tf.constant(RNA), "atac_matrix": tf.constant(ATAC)})
        atac_mod = (ATAC + 1) * 10
        rna_mod = (RNA + 1) * 2 - 3
        expected = np.concatenate(
            [atac_mod.sum(-1, keepdims=True), rna_mod.sum(-1, keepdims=True)], axis=-1
        )
        np.testing.assert_allclose(out[X].numpy(), expected, rtol=1e-6)

    def test_single_modality_without_libsize(self, modifier_lookup):
        model = Preprocessor.make(
            ["cnv"], {"cnv": "Studentt"}, {"cnv": 3}, n_dims=4, name="pre"
        )
        modifier_lookup.assert_called_once_with("Studentt", "modules/preprocessors/modifiers", "Modifier")
        assert set(model.output) == {X}
        _set_dense(model, "pre_cnv", 1.0)
        out = model({"cnv_matrix": tf.constant(ATAC)})
        np.testing.assert_allclose(
            out[X].numpy(), np.repeat((ATAC + 1).sum(-1, keepdims=True), 4, -1), rtol=1e-6
        )

    def test_default_n_dims(self, modifier_lookup):
        model = Preprocessor.make(["cnv"], {"cnv": "MultivariateNormalDiag"}, {"cnv": 3})
        assert model.name == "preprocessor"
        assert tuple(model.output[X].shape) == (None, 1024)

    def test_train_step_not_implemented(self, modifier_lookup):
        model = Preprocessor.make(["cnv"], {"cnv": "Studentt"}, {"cnv": 3}, n_dims=2)
        with pytest.raises(NotImplementedError):
            model.train_step(None)
