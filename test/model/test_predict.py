"""Model.predict (cavachon/model/model.py).

Mocked: DataLoader (MagicMock yielding hand-built batches), Component
(FakeComponent), GeneralUtils, ComponentConfig (FakeConfig). muon/anndata
objects are real, built in memory (no file I/O).
"""

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from test.model.conftest import FakeConfig

N_OBS = 5


def _mdata():
    rng = np.random.default_rng(1)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)])
    rna = ad.AnnData(rng.integers(0, 5, (N_OBS, 2)).astype(np.float32), obs=obs.copy())
    atac = ad.AnnData(rng.integers(0, 2, (N_OBS, 3)).astype(np.float32), obs=obs.copy())
    return mu.MuData({"rna": rna, "atac": atac})


def _batches(mdata, sizes=(3, 2), modalities=("rna", "atac")):
    out, start = [], 0
    for s in sizes:
        b = {}
        for m in modalities:
            b[f"{m}_matrix"] = tf.constant(mdata.mod[m].X[start : start + s])
            b[f"{m}_batch_effect"] = tf.zeros((s, 1))
        out.append(b)
        start += s
    return out


@pytest.fixture
def setup(make_model, fakes):
    a = FakeConfig("a", ["rna"], kernel_value=0.5, z_hat_offset=10.0)
    b = FakeConfig(
        "b",
        ["rna", "atac"],
        n_vars={"rna": 2, "atac": 3},
        kernel_value=1.0,
        z_hat_offset=20.0,
        conditioned_on_z_hat=["a"],
        save_z={"rna": False, "atac": True},
        save_x={"rna": True, "atac": False},
    )
    model = make_model(a, b)
    mdata = _mdata()
    fakes.dataloader.return_value = _batches(mdata)
    return model, mdata, fakes


def _expected(mdata):
    x_rna, x_atac = mdata.mod["rna"].X, mdata.mod["atac"].X
    a_z = x_rna @ np.full((2, 2), 0.5)
    b_z = x_rna @ np.full((2, 2), 1.0) + x_atac @ np.full((3, 2), 1.0)
    return {
        "a_z": a_z,
        "a_z_hat": a_z + 10,
        "a_rna_x_parameters": np.repeat((a_z + 10).sum(-1, keepdims=True), 2, -1),
        "b_z": b_z,
        "b_z_hat": b_z + 20,
        "b_rna_x_parameters": np.repeat((b_z + 20).sum(-1, keepdims=True), 2, -1),
    }


class TestPredictMuData:
    def test_dataloader_built_with_mdata_and_batch_size(self, setup):
        model, mdata, fakes = setup
        model.predict(mdata, batch_size=7)
        fakes.dataloader.assert_called_once_with(mdata, batch_size=7)

    def test_default_batch_size_is_one(self, setup):
        model, mdata, fakes = setup
        model.predict(mdata)
        fakes.dataloader.assert_called_once_with(mdata, batch_size=1)

    def test_outputs_concatenated_in_batch_order(self, setup):
        model, mdata, _ = setup
        outputs = model.predict(mdata, batch_size=3)
        for key, value in _expected(mdata).items():
            assert outputs[key].shape[0] == N_OBS
            np.testing.assert_allclose(outputs[key], value, rtol=1e-6, err_msg=key)

    def test_obsm_written_per_save_flags(self, setup):
        model, mdata, _ = setup
        model.predict(mdata, batch_size=3)
        exp = _expected(mdata)
        rna, atac = mdata.mod["rna"].obsm, mdata.mod["atac"].obsm
        assert set(rna.keys()) == {
            "z_a",
            "z_hat_a",
            "x_parameters_a",
            "x_parameters_b",
        }
        assert set(atac.keys()) == {"z_b", "z_hat_b"}
        np.testing.assert_allclose(rna["z_a"], exp["a_z"], rtol=1e-6)
        np.testing.assert_allclose(rna["z_hat_a"], exp["a_z_hat"], rtol=1e-6)
        np.testing.assert_allclose(
            rna["x_parameters_a"], exp["a_rna_x_parameters"], rtol=1e-6
        )
        np.testing.assert_allclose(
            rna["x_parameters_b"], exp["b_rna_x_parameters"], rtol=1e-6
        )
        np.testing.assert_allclose(atac["z_b"], exp["b_z"], rtol=1e-6)
        np.testing.assert_allclose(atac["z_hat_b"], exp["b_z_hat"], rtol=1e-6)

    def test_returns_outputs_dict(self, setup):
        model, mdata, _ = setup
        outputs = model.predict(mdata)
        assert {
            "a_z",
            "a_z_hat",
            "b_z",
            "b_z_hat",
            "a_rna_x_parameters",
            "b_rna_x_parameters",
        } <= set(outputs)
        assert all(isinstance(v, np.ndarray) for v in outputs.values())

    def test_inference_mode(self, setup):
        model, mdata, _ = setup
        model.predict(mdata)
        assert model.components["a"].recorder.calls[-1]["training"] is False

    def test_no_z_written_when_save_z_false(self, make_model, fakes):
        model = make_model(
            FakeConfig("a", ["rna"], save_z={"rna": False}, save_x={"rna": False})
        )
        mdata = _mdata()
        fakes.dataloader.return_value = _batches(mdata, modalities=("rna",))
        outputs = model.predict(mdata)
        assert len(mdata.mod["rna"].obsm.keys()) == 0
        # z / z_hat are still computed and returned
        assert set(outputs) == {"a_z", "a_z_hat"}

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: predict_x is sticky across modalities; x_parameters collected for save_x=False modalities (order-dependent)",
    )
    def test_x_parameters_only_for_save_x_modalities(self, setup):
        model, mdata, _ = setup
        outputs = model.predict(mdata)
        # b: save_x = {rna: True, atac: False}
        assert "b_atac_x_parameters" not in outputs

    def test_unused_modality_in_mudata_is_ignored(self, make_model, fakes):
        # regression: fixed bug, see UNIT_TEST_REPORT.md
        # Model only uses rna, the MuData (and so DataLoader batches) also
        # carry atac. Keras maps dict inputs positionally when the keys do
        # not match, so atac_matrix is fed as rna_matrix.
        model = make_model(FakeConfig("a", ["rna"], kernel_value=0.5))
        mdata = _mdata()
        fakes.dataloader.return_value = _batches(mdata, modalities=("rna", "atac"))
        outputs = model.predict(mdata)
        np.testing.assert_allclose(
            outputs["a_z"], mdata.mod["rna"].X @ np.full((2, 2), 0.5), rtol=1e-6
        )


class TestPredictTensors:
    def test_mapping_input_delegates_to_keras_predict(self, make_model):
        # regression: fixed bug, see UNIT_TEST_REPORT.md
        model = make_model(FakeConfig("a", ["rna"]))
        batch = {
            "rna_matrix": np.ones((4, 2), np.float32),
            "rna_batch_effect": np.zeros((4, 1), np.float32),
        }
        try:
            out = model.predict(batch, batch_size=2, verbose=0)
        except AttributeError as e:  # surface the bug as an assertion failure
            raise AssertionError(f"predict(Mapping) crashed: {e}")
        np.testing.assert_allclose(out["a_z"], np.full((4, 2), 1.0))
