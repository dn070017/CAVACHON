"""Model.train_step (cavachon/model/model.py).

Mocked: losses (FakeStandardKL / FakeGMMKL / FakeNLL / FakeGMMDensity,
value = weight_factor * reduce_sum(y_pred)), ProgressiveScaler (FakeScaler),
ToDense (FakeToDense), TensorUtils.remove_nan_gradients (MagicMock),
Component (FakeComponent), GeneralUtils. The optimizer is a real Keras SGD.

Reference numbers for one component ``a`` (modality rna, n_vars=2,
latent=2, kernel W=0.5, z_hat_offset=10, batch of B=3 rows ``x``):

    S          = sum(x @ W)
    GMM KL raw = sum([z, 2z])                 = 3 S
    NLL raw    = sum(rowsum(z_hat) * 1_2)     = 2 (S + 10 * B * 2) = 2 S + 120
    dS/dW_ij   = colsum(x)_i
"""

import numpy as np
import pytest
import tensorflow as tf

from test.model.conftest import (
    FakeConfig,
    FakeGMMDensity,
    FakeToDense,
    PlainLoss,
    make_batch,
)

STD = "standard_kl_divergence"
GMM = "gmm_kl_divergence"
NLL = "negative_log_data_likelihood"
ZD = "z_hat_gmm_density"
ZKL = "z_hat_gmm_kl_divergence"


def _S(x, w=0.5, latent=2):
    return float((x @ np.full((x.shape[1], latent), w)).sum())


@pytest.fixture
def single(make_model):
    m = make_model(FakeConfig("a", ["rna"]))
    batch = make_batch({"rna": 2})
    x = batch["rna_matrix"].numpy()
    return m, batch, x, _S(x)


def _f(d):
    return {k: float(v) for k, v in d.items()}


class TestLossAggregation:
    def test_default_total_and_metrics(self, single):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        out = _f(m.train_step(batch))
        assert out == pytest.approx(
            {
                "loss": 5 * S + 120,
                f"a_{GMM}": 3 * S,
                f"a_rna_{NLL}": 2 * S + 120,
            }
        )

    def test_y_true_y_pred_routing(self, single):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m.train_step(batch)
        z = x @ np.full((2, 2), 0.5)
        gmm = m.loss[f"a_{GMM}"]
        y_true, y_pred = gmm.calls[0]
        np.testing.assert_allclose(y_true, np.full((3, 2), 7.0))  # prior params
        np.testing.assert_allclose(y_pred, np.concatenate([z, 2 * z], -1))
        nll = m.loss[f"a_rna_{NLL}"]
        y_true, y_pred = nll.calls[0]
        np.testing.assert_allclose(y_true, x)
        np.testing.assert_allclose(
            y_pred, np.repeat((z + 10).sum(-1, keepdims=True), 2, -1)
        )

    def test_nll_target_goes_through_to_dense(self, single):
        m, batch, x, S = single
        FakeToDense.transform = staticmethod(lambda v: v + 1000.0)
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m.train_step(batch)
        assert [t.key for t in FakeToDense.instances] == ["rna_matrix"]
        y_true, _ = m.loss[f"a_rna_{NLL}"].calls[0]
        np.testing.assert_allclose(y_true, x + 1000.0)

    def test_each_loss_counted_once_in_total(self, make_model):
        a = FakeConfig("a", ["rna"])
        b = FakeConfig(
            "b", ["rna", "atac"], n_vars={"rna": 2, "atac": 3}, conditioned_on_z=["a"]
        )
        m = make_model(a, b)
        m.compile(
            standard_kl_weights={"a": 0.5, "b": 1.5},
            gmm_kl_weights={"a": 2.0, "b": 1.0},
            optimizer=tf.keras.optimizers.SGD(0.0),
        )
        batch = make_batch({"rna": 2, "atac": 3})
        out = _f(m.train_step(batch))
        weighted = {}
        for key, fn in m.loss.items():
            # the first call of each loss is the one that enters the total
            y_true, y_pred = fn.calls[0]
            weighted[key] = float(fn.weight.factor * tf.reduce_sum(y_pred))
            # loss evaluated twice: once for the total, once for metrics
            assert len(fn.calls) == 2
        assert len(weighted) == 7  # 2x(std+gmm) + 3 NLL
        assert out["loss"] == pytest.approx(sum(weighted.values()))

    def test_both_kl_losses_get_same_inputs(self, single):
        m, batch, x, S = single
        m.compile(
            standard_kl_weights={"a": 0.5},
            gmm_kl_weights={"a": 2.0},
            optimizer=tf.keras.optimizers.SGD(0.0),
        )
        out = _f(m.train_step(batch))
        std, gmm = m.loss[f"a_{STD}"], m.loss[f"a_{GMM}"]
        np.testing.assert_allclose(std.calls[0][0], gmm.calls[0][0])
        np.testing.assert_allclose(std.calls[0][1], gmm.calls[0][1])
        assert out["loss"] == pytest.approx(0.5 * 3 * S + 2.0 * 3 * S + 2 * S + 120)
        # per-loss metrics are unweighted
        assert out[f"a_{STD}"] == pytest.approx(3 * S)
        assert out[f"a_{GMM}"] == pytest.approx(3 * S)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: standard-KL-only component looks up the GMM key, so its KL is never trained",
    )
    def test_standard_kl_only_is_trained(self, single):
        m, batch, x, S = single
        m.compile(
            standard_kl_weights={"a": 1.0}, optimizer=tf.keras.optimizers.SGD(0.0)
        )
        assert set(m.loss) == {f"a_{STD}", f"a_rna_{NLL}"}
        out = _f(m.train_step(batch))
        assert len(m.loss[f"a_{STD}"].calls) > 0
        assert out["loss"] == pytest.approx(5 * S + 120)
        assert out[f"a_{STD}"] == pytest.approx(3 * S)

    def test_regularization_losses_added(self, make_model):
        m = make_model(
            FakeConfig("a", ["rna"], kernel_regularizer=tf.keras.regularizers.L2(0.5))
        )
        batch = make_batch({"rna": 2})
        S = _S(batch["rna_matrix"].numpy())
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        out = _f(m.train_step(batch))
        reg = 0.5 * 4 * 0.25
        assert out["loss"] == pytest.approx(5 * S + 120 + reg)

    def test_components_called_with_training_true(self, single):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m.train_step(batch)
        assert m.components["a"].recorder.calls[-1]["training"] is True

    def test_custom_loss_dict_only_uses_present_keys(self, single):
        m, batch, x, S = single
        custom = PlainLoss(factor=2.0)
        with pytest.warns(RuntimeWarning):
            m.compile(
                loss={f"a_rna_{NLL}": custom}, optimizer=tf.keras.optimizers.SGD(0.0)
            )
        out = _f(m.train_step(batch))
        # no weight attribute -> metric is the raw (already weighted) value
        assert out == pytest.approx(
            {"loss": 2 * (2 * S + 120), f"a_rna_{NLL}": 2 * (2 * S + 120)}
        )


class TestWeightedMetrics:
    def test_nll_loss_weight_scales_total_not_metric(self, single):
        m, batch, x, S = single
        m.compile(
            loss_weights={f"a_rna_{NLL}": 2.0}, optimizer=tf.keras.optimizers.SGD(0.0)
        )
        out = _f(m.train_step(batch))
        assert out["loss"] == pytest.approx(3 * S + 2 * (2 * S + 120))
        assert out[f"a_rna_{NLL}"] == pytest.approx(2 * S + 120)

    def test_scheduler_pinned_scaler_affects_total_only(self, single):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m._gmm_kl_weights["a"].pin_to(0.25)
        out = _f(m.train_step(batch))
        assert out["loss"] == pytest.approx(0.25 * 3 * S + 2 * S + 120)
        assert out[f"a_{GMM}"] == pytest.approx(3 * S)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: scaler-weighted metric divides by the weight without a zero guard (NaN)",
    )
    def test_zero_scaler_weight_metric_is_finite(self, single):
        m, batch, x, S = single
        # weight 0 is accepted by compile (loss is created with scale 0)
        m.compile(
            standard_kl_weights={"a": 0.0},
            gmm_kl_weights={"a": 1.0},
            optimizer=tf.keras.optimizers.SGD(0.0),
        )
        out = _f(m.train_step(batch))
        assert np.isfinite(out[f"a_{STD}"])


class TestZHatLosses:
    @pytest.fixture
    def zmodel(self, make_model):
        m = make_model(FakeConfig("a", ["rna"], learn_z_hat_priors=True))
        batch = make_batch({"rna": 2})
        x = batch["rna_matrix"].numpy()
        return m, batch, x, _S(x)

    def test_z_hat_routing_and_weights(self, zmodel):
        m, batch, x, S = zmodel
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m._z_hat_density_weights["a"].assign(2.0)
        m._z_hat_gmm_kl_weights["a"].pin_to(0.5)
        out = _f(m.train_step(batch))

        z = x @ np.full((2, 2), 0.5)
        z_hat = z + 10.0
        dens = m.loss[f"a_{ZD}"]
        np.testing.assert_allclose(dens.calls[0][0], np.full((3, 2), 11.0))
        np.testing.assert_allclose(dens.calls[0][1], z_hat)
        zkl = m.loss[f"a_{ZKL}"]
        np.testing.assert_allclose(zkl.calls[0][0], np.full((3, 2), 11.0))
        np.testing.assert_allclose(
            zkl.calls[0][1], np.concatenate([z_hat, 3 * z_hat], -1)
        )

        sum_zh = z_hat.sum()
        assert out["loss"] == pytest.approx(
            5 * S + 120 + 2.0 * sum_zh + 0.5 * 4 * sum_zh, rel=1e-5
        )
        assert out[f"a_{ZD}"] == pytest.approx(sum_zh, rel=1e-5)
        assert out[f"a_{ZKL}"] == pytest.approx(4 * sum_zh, rel=1e-5)

    def test_density_metric_zero_when_weight_zero(self, zmodel):
        m, batch, x, S = zmodel
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        m._z_hat_density_weights["a"].assign(0.0)
        m._z_hat_gmm_kl_weights["a"].pin_to(0.5)
        out = _f(m.train_step(batch))
        assert out[f"a_{ZD}"] == 0.0

    def test_no_z_hat_losses_called_when_not_compiled(self, single):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(0.0))
        out = m.train_step(batch)
        assert f"a_{ZD}" not in out and f"a_{ZKL}" not in out
        assert FakeGMMDensity.instances == []


class TestGradients:
    def test_gradients_hand_computed_and_applied(self, single, fakes):
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
        m.train_step(batch)
        kernel = m.components["a"].encoders["rna"].kernel
        colsum = x.sum(0)
        expected_grad = 5 * np.repeat(colsum[:, None], 2, axis=1)
        np.testing.assert_allclose(kernel.numpy(), 0.5 - 0.5 * expected_grad, rtol=1e-6)

    def test_remove_nan_gradients_output_is_applied(self, single, fakes):
        m, batch, x, S = single
        received = []

        def replace(grads):
            received.append([g.numpy() for g in grads])
            return [tf.fill(g.shape, 0.2) for g in grads]

        fakes.tensor_utils.remove_nan_gradients.side_effect = replace
        m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
        m.train_step(batch)
        fakes.tensor_utils.remove_nan_gradients.assert_called_once()
        assert len(received[0]) == 1
        np.testing.assert_allclose(
            received[0][0], 5 * np.repeat(x.sum(0)[:, None], 2, axis=1), rtol=1e-6
        )
        kernel = m.components["a"].encoders["rna"].kernel
        np.testing.assert_allclose(kernel.numpy(), np.full((2, 2), 0.5 - 0.5 * 0.2))

    def test_frozen_component_not_updated(self, make_model, fakes):
        a = FakeConfig("a", ["rna"])
        b = FakeConfig("b", ["rna"], conditioned_on_z=["a"])
        m = make_model(a, b)
        m.components["b"].trainable = False
        m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))
        m.train_step(make_batch({"rna": 2}))
        (grads,), _ = fakes.tensor_utils.remove_nan_gradients.call_args
        assert len(grads) == 1
        np.testing.assert_array_equal(
            m.components["b"].encoders["rna"].kernel.numpy(), np.full((2, 2), 0.5)
        )
        assert not np.allclose(
            m.components["a"].encoders["rna"].kernel.numpy(), np.full((2, 2), 0.5)
        )


class TestFitPath:
    def test_fit_runs_traced_train_step(self, single):
        """Production path: scheduler calls model.fit (tf.function)."""
        m, batch, x, S = single
        m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5))
        ds = tf.data.Dataset.from_tensors(batch)
        history = m.fit(ds, epochs=1, verbose=0)
        assert set(history.history) == {"loss", f"a_{GMM}", f"a_rna_{NLL}"}
        assert history.history["loss"][0] == pytest.approx(5 * S + 120, rel=1e-5)
        kernel = m.components["a"].encoders["rna"].kernel.numpy()
        expected = 0.5 - 0.5 * 5 * np.repeat(x.sum(0)[:, None], 2, axis=1)
        np.testing.assert_allclose(kernel, expected, rtol=1e-6)
