"""Model.encode / hierarchical_encode / decode (cavachon/model/model.py).

Mocked: Component (FakeComponent with recording encode /
hierarchical_encode / decode), GeneralUtils, ComponentConfig (FakeConfig).
"""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from test.model.conftest import FakeConfig, make_batch


@pytest.fixture
def model(make_model):
    a = FakeConfig("a", ["rna"], kernel_value=0.5, z_hat_offset=10.0)
    b = FakeConfig(
        "b",
        ["atac"],
        n_vars={"atac": 3},
        kernel_value=1.0,
        z_hat_offset=20.0,
        conditioned_on_z=["a"],
        conditioned_on_z_hat=["a"],
    )
    return make_model(a, b)


@pytest.fixture
def batch():
    return make_batch({"rna": 2, "atac": 3})


def _rec(model, name):
    return model.components[name].recorder


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


class TestEncode:
    def test_unknown_component_raises(self, model, batch):
        with pytest.raises(ValueError, match=r"Unknown component names: \['x', 'y'\]"):
            model.encode(batch, components=["a", "y", "x"])

    def test_all_components_structure_and_values(self, model, batch):
        out = model.encode(batch)
        assert set(out) == {"z_parameters", "z"}
        assert set(out["z"]) == {"a", "b"} == set(out["z_parameters"])
        z_a = batch["rna_matrix"].numpy() @ np.full((2, 2), 0.5)
        np.testing.assert_allclose(out["z"]["a"], z_a)
        np.testing.assert_allclose(out["z_parameters"]["a"], 2 * z_a)

    def test_subset_and_no_conditionals_in_batch(self, model, batch):
        out = model.encode(batch, components=["b"])
        assert set(out["z"]) == {"b"}
        assert _rec(model, "a").encode_calls == []
        (call,) = _rec(model, "b").encode_calls
        # b is conditioned on a, but encode() strips the conditionals
        assert set(call["batch"]) == {"atac_matrix", "atac_batch_effect"}
        assert call["batch"]["atac_matrix"] is batch["atac_matrix"]

    @pytest.mark.parametrize(
        "training,trainable,expected",
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_training_flag(self, model, batch, training, trainable, expected):
        model.components["a"].trainable = trainable
        model.encode(batch, components=["a"], training=training)
        assert _rec(model, "a").encode_calls[-1]["training"] is expected


# ---------------------------------------------------------------------------
# hierarchical_encode
# ---------------------------------------------------------------------------


class TestHierarchicalEncode:
    def _z(self):
        return {"a": tf.fill((3, 2), 1.0), "b": tf.fill((3, 2), 2.0)}

    def test_unknown_component_raises(self, model, batch):
        with pytest.raises(ValueError, match="Unknown component names"):
            model.hierarchical_encode(batch, self._z(), components=["nope"])

    def test_missing_z_raises(self, model, batch):
        with pytest.raises(ValueError, match="Missing z for component 'b'"):
            model.hierarchical_encode(batch, {"a": tf.ones((3, 2))})

    def test_accumulates_parent_z_hat_and_passes_parent_z(self, model, batch):
        z = self._z()
        out = model.hierarchical_encode(batch, z)
        assert set(out) == {"z_hat"}
        np.testing.assert_allclose(out["z_hat"]["a"], np.full((3, 2), 11.0))
        np.testing.assert_allclose(out["z_hat"]["b"], np.full((3, 2), 22.0))
        call_b = _rec(model, "b").hierarchical_encode_calls[-1]
        np.testing.assert_allclose(call_b["z"], z["b"])
        np.testing.assert_allclose(call_b["batch"]["z_conditional"], z["a"])
        # z_hat_conditional is a's freshly computed z_hat, not its z
        np.testing.assert_allclose(
            call_b["batch"]["z_hat_conditional"], np.full((3, 2), 11.0)
        )
        assert {"atac_matrix", "atac_batch_effect"} <= set(call_b["batch"])

    def test_seed_supplies_parent_z_hat(self, model, batch):
        seed = {"a": tf.fill((3, 2), -5.0)}
        out = model.hierarchical_encode(
            batch, self._z(), z_hat_seed=seed, components=["b"]
        )
        assert set(out["z_hat"]) == {"b"}
        call_b = _rec(model, "b").hierarchical_encode_calls[-1]
        np.testing.assert_allclose(
            call_b["batch"]["z_hat_conditional"], np.full((3, 2), -5.0)
        )
        assert _rec(model, "a").hierarchical_encode_calls == []
        # seed dict is copied, not mutated
        assert set(seed) == {"a"}

    def test_computed_z_hat_overrides_seed(self, model, batch):
        seed = {"a": tf.fill((3, 2), -5.0)}
        model.hierarchical_encode(batch, self._z(), z_hat_seed=seed)
        call_b = _rec(model, "b").hierarchical_encode_calls[-1]
        np.testing.assert_allclose(
            call_b["batch"]["z_hat_conditional"], np.full((3, 2), 11.0)
        )

    def test_strict_missing_parent_z_raises(self, model, batch):
        with pytest.raises(ValueError, match="parent z for component 'b': 'a'"):
            model.hierarchical_encode(batch, {"b": tf.ones((3, 2))}, components=["b"])

    def test_strict_missing_parent_z_hat_raises(self, model, batch):
        with pytest.raises(ValueError, match="parent z_hat for component 'b': 'a'"):
            model.hierarchical_encode(batch, self._z(), components=["b"])

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: strict=False skips the parent but prepare_component_inputs still concats None",
    )
    def test_non_strict_skips_missing_parent(self, model, batch):
        out = model.hierarchical_encode(
            batch, {"b": tf.ones((3, 2))}, components=["b"], strict=False
        )
        np.testing.assert_allclose(out["z_hat"]["b"], np.full((3, 2), 21.0))

    @pytest.mark.parametrize(
        "training,trainable,expected",
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_training_flag(self, model, batch, training, trainable, expected):
        model.components["a"].trainable = trainable
        model.hierarchical_encode(batch, self._z(), components=["a"], training=training)
        assert _rec(model, "a").hierarchical_encode_calls[-1]["training"] is expected


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


class TestDecode:
    def test_unknown_component_raises(self, model, batch):
        with pytest.raises(ValueError, match="Unknown component names"):
            model.decode(batch, {}, components=["zzz"])

    def test_missing_z_hat_strict_raises(self, model, batch):
        with pytest.raises(ValueError, match="Missing z_hat for component 'b'"):
            model.decode(batch, {"a": tf.ones((3, 2))})

    def test_missing_z_hat_non_strict_skips(self, model, batch):
        out = model.decode(batch, {"a": tf.ones((3, 2))}, strict=False)
        assert set(out["x_parameters"]) == {"a_rna_x_parameters"}
        assert _rec(model, "b").decode_calls == []

    def test_outputs_and_batch(self, model, batch):
        z_hat = {"a": tf.fill((3, 2), 1.0), "b": tf.fill((3, 2), 2.0)}
        out = model.decode(batch, z_hat)
        assert set(out) == {Constants.MODEL_OUTPUTS_X_PARAMS}
        assert set(out["x_parameters"]) == {"a_rna_x_parameters", "b_atac_x_parameters"}
        np.testing.assert_allclose(
            out["x_parameters"]["a_rna_x_parameters"], np.full((3, 2), 2.0)
        )
        np.testing.assert_allclose(
            out["x_parameters"]["b_atac_x_parameters"], np.full((3, 3), 4.0)
        )
        call_b = _rec(model, "b").decode_calls[-1]
        assert call_b["z_hat"] is z_hat["b"]
        assert set(call_b["batch"]) == {"atac_matrix", "atac_batch_effect"}

    @pytest.mark.parametrize(
        "training,trainable,expected",
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_training_flag(self, model, batch, training, trainable, expected):
        model.components["b"].trainable = trainable
        model.decode(batch, {"b": tf.ones((3, 2))}, components=["b"], training=training)
        assert _rec(model, "b").decode_calls[-1]["training"] is expected
