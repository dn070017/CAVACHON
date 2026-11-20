"""KL annealing schedule, GMMPriorInitializer (k-means++ prior
initialisation), the compile helper and mlflow runs."""

import numpy as np
import pytest
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.scheduler.gmm_prior_initializer import GMMPriorInitializer
from cavachon.scheduler.phase_planner import PhasePlanner
from cavachon.scheduler.sequential_training_scheduler import STANDARD_KL_WEIGHT
from test.scheduler.conftest import FakeVar, make_config, make_scheduler

STD = f"A_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"
GMM = f"A_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"
DENS = f"A_{Constants.MODEL_LOSS_Z_HAT_GMM_DENSITY_POSTFIX}"


def softplus(x):
    return np.log1p(np.exp(x))


# ---------------------------------------------------------------------------
# PhasePlanner.make_kl_schedule
# ---------------------------------------------------------------------------


class TestKLSchedule:
    def test_three_stages_default_ratio(self):
        schedule = PhasePlanner.make_kl_schedule("A", 10, (0.5, 0.2, 0.3))
        values = [schedule(e) for e in range(10)]
        # epochs 0-4: standard KL only
        for e in range(5):
            assert values[e] == {STD: STANDARD_KL_WEIGHT, GMM: 0.0, DENS: 0.0}
        # epochs 5-6: linear crossfade, t = 0, 0.5
        assert values[5][STD] == pytest.approx(1.0)
        assert values[5][GMM] == pytest.approx(0.0)
        assert values[6][STD] == pytest.approx(0.5)
        assert values[6][GMM] == pytest.approx(0.5)
        # epochs 7-9: GMM only
        for e in range(7, 10):
            assert values[e] == {STD: 0.0, GMM: 1.0, DENS: 0.0}

    def test_weights_sum_to_one_during_crossfade(self):
        schedule = PhasePlanner.make_kl_schedule("A", 20, (0.25, 0.5, 0.25))
        for e in range(20):
            v = schedule(e)
            assert v[STD] + v[GMM] == pytest.approx(1.0)
            assert v[DENS] == 0.0

    def test_zero_crossfade_does_not_divide_by_zero(self):
        schedule = PhasePlanner.make_kl_schedule("A", 4, (0.5, 0.0, 0.5))
        assert [schedule(e)[GMM] for e in range(4)] == [0.0, 0.0, 1.0, 1.0]

    def test_crossfade_boundary_robust_to_float_rounding(self):
        # regression: int() truncation of n*(r0+r1) (0.1+0.7=0.7999..) moved
        # the GMM start one epoch early and cut the crossfade short
        schedule = PhasePlanner.make_kl_schedule("A", 10, (0.1, 0.7, 0.2))
        # standard KL ends at epoch 1, GMM-only should start at epoch 8:
        # epoch 7 is the last crossfade epoch with t = 6/7
        assert schedule(7)[GMM] == pytest.approx(6 / 7)
        assert schedule(7)[STD] == pytest.approx(1 / 7)

    def test_constant_schedule_changes_nothing(self):
        assert PhasePlanner.constant_schedule(0) == {}
        assert PhasePlanner.constant_schedule(99) == {}


# ---------------------------------------------------------------------------
# GMMPriorInitializer: k-means++ and the estimate
# ---------------------------------------------------------------------------


class TestKMeansPlusPlus:
    def test_picks_one_center_per_separated_cluster(self):
        X = tf.constant(
            [[0.0, 0.0], [0.1, 0.0], [100.0, 100.0], [100.1, 100.0], [-100.0, 50.0]],
            dtype=tf.float32,
        )
        centers = GMMPriorInitializer.kmeans_plus_plus(X, 3, seed=1).numpy()
        assert centers.shape == (3, 2)
        # every center is a data point, and they are all in distinct clusters
        groups = {
            int(
                np.argmin(
                    [np.linalg.norm(c - g) for g in ([0, 0], [100, 100], [-100, 50])]
                )
            )
            for c in centers
        }
        assert groups == {0, 1, 2}

    def test_cluster_assignments_are_nearest_center(self):
        X = tf.constant([[0.0], [0.9], [2.1], [3.0]], dtype=tf.float32)
        centers = tf.constant([[0.0], [3.0]], dtype=tf.float32)
        out = GMMPriorInitializer.assign_clusters(X, centers).numpy()
        assert out.tolist() == [0, 0, 1, 1]
        assert out.dtype == np.int32

    def test_fewer_distinct_points_than_clusters(self):
        # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
        X = tf.constant([[0.0, 0.0]] * 5 + [[1.0, 1.0]] * 5, dtype=tf.float32)
        centers = GMMPriorInitializer.kmeans_plus_plus(X, 3, seed=0)
        assert centers.shape == (3, 2)


def _two_cluster_data(std, n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-100.0, scale=std, size=(n, 2))
    b = rng.normal(loc=+100.0, scale=std, size=(n // 4, 2))
    return np.concatenate([a, b]).astype(np.float32)


def _initializer(model=None, add_noise=False):
    return GMMPriorInitializer(
        model, "MDATA", 17, ["A", "B"], add_noise=add_noise, seed=42
    )


class TestInverseSoftplus:
    @pytest.mark.parametrize("std", [1e-3, 0.5, 1.0, 3.0, 10.0, 100.0, 500.0])
    def test_softplus_recovers_std(self, std):
        bias = GMMPriorInitializer.inverse_softplus(np.array([std]))
        assert bias.dtype == np.float32
        assert np.all(np.isfinite(bias))  # no overflow for large std
        np.testing.assert_allclose(softplus(bias.astype(np.float64)), [std], rtol=1e-5)

    def test_zero_std_is_floored(self):
        bias = GMMPriorInitializer.inverse_softplus(np.array([0.0, 2.0]))
        assert np.all(np.isfinite(bias))
        assert softplus(float(bias[0])) == pytest.approx(1e-6, rel=1e-3)


class TestEstimate:
    def test_means_and_log_proportions(self):
        z = _two_cluster_data(0.5)
        estimate = _initializer().estimate(z, 2)
        means = sorted(float(m) for m in np.asarray(estimate.locs)[:, 0])
        assert means[0] == pytest.approx(-100.0, abs=3)
        assert means[1] == pytest.approx(100.0, abs=3)
        # logits = log of cluster proportions (400 vs 100 points)
        probs = sorted(np.exp(estimate.logits.ravel()))
        assert probs == [pytest.approx(0.2), pytest.approx(0.8)]
        assert estimate.logits.shape == (1, 2)
        assert estimate.scale_biases.shape == (2, 2)

    def test_noise_moves_the_centers_off_the_data_points(self):
        z = _two_cluster_data(0.5)

        def distance_to_nearest_point(locs):
            locs = np.asarray(locs)
            return np.linalg.norm(z[None, :, :] - locs[:, None, :], axis=2).min(axis=1)

        plain = _initializer(add_noise=False).estimate(z, 2)
        noisy = _initializer(add_noise=True).estimate(z, 2)
        # k-means++ picks data points; the noise (std 0.1) moves them a bit
        np.testing.assert_array_equal(distance_to_nearest_point(plain.locs), 0.0)
        d = distance_to_nearest_point(noisy.locs)
        assert np.all(d > 0) and np.all(d < 1.0)

    def test_apply_writes_every_parameter(self, fakes):
        _, model = make_scheduler([make_config("A")])
        prior = model.components["A"].z_prior_parameterizer
        z = _two_cluster_data(0.5)
        estimate = _initializer().estimate(z, 2)
        GMMPriorInitializer.apply(prior, estimate)
        for k in range(2):
            np.testing.assert_allclose(
                prior.loc_bias[k].numpy(), np.asarray(estimate.locs)[k : k + 1]
            )
            np.testing.assert_allclose(
                prior.scale_diag_bias[k].numpy(), estimate.scale_biases[k : k + 1]
            )
        np.testing.assert_allclose(prior.logits_bias.numpy(), estimate.logits)


# ---------------------------------------------------------------------------
# GMMPriorInitializer.initialize (with the model)
# ---------------------------------------------------------------------------


class TestInitialize:
    def _run(self, std, target="z"):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs, learn_z_hat=("B",), batch_size=17)
        z = _two_cluster_data(std)
        model.predict_outputs = {"B_z": z, "B_z_hat": z + 1000.0}
        # progressive state that must survive the k-means call
        model.components["B"].set_progressive_scaler_iteration(5.0, 30.0)
        s.prior_initializer.add_noise = False
        out = s.prior_initializer.initialize("B", target=target)
        return s, model, z, out

    @staticmethod
    def _prior_std_vs_cluster_std(model, z):
        prior = model.components["B"].z_prior_parameterizer
        got, expected = [], []
        for loc, bias in zip(prior.loc_bias, prior.scale_diag_bias):
            members = z[np.sign(z[:, 0]) == np.sign(loc.numpy()[0, 0])]
            got.append(softplus(bias.numpy()[0]))
            expected.append(members.std(axis=0))
        return np.array(got), np.array(expected)

    def test_writes_means_logits_and_returns_z(self, fakes):
        s, model, z, out = self._run(0.5)
        prior = model.components["B"].z_prior_parameterizer
        np.testing.assert_array_equal(out, z)
        means = sorted(float(v.numpy()[0, 0]) for v in prior.loc_bias)
        assert means[0] == pytest.approx(-100.0, abs=3)
        assert means[1] == pytest.approx(100.0, abs=3)
        probs = sorted(np.exp(prior.logits_bias.numpy().ravel()))
        assert probs == [pytest.approx(0.2), pytest.approx(0.8)]
        assert model.predict_args == ("MDATA", 17, 0)
        # untouched: z_hat prior
        assert all(
            np.all(v.numpy() == 0)
            for v in model.components["B"].z_hat_prior_parameterizer.loc_bias
        )

    def test_z_hat_target(self, fakes):
        s, model, z, out = self._run(0.5, target="z_hat")
        prior = model.components["B"].z_hat_prior_parameterizer
        np.testing.assert_array_equal(out, z + 1000.0)
        means = sorted(float(v.numpy()[0, 0]) for v in prior.loc_bias)
        assert means[0] == pytest.approx(900.0, abs=3)
        assert all(
            np.all(v.numpy() == 0)
            for v in model.components["B"].z_prior_parameterizer.loc_bias
        )

    def test_freezes_for_prediction_and_restores_progressive_state(self, fakes):
        s, model, _, _ = self._run(0.5)
        assert model.trainable_history == [False, True]
        assert model.components["B"].progressive == (5.0, 30.0)
        assert model.components["A"].progressive == (1.0, 1.0)

    def test_restores_every_nested_trainable_flag(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs)
        model.predict_outputs = {"A_z": _two_cluster_data(0.5)}
        a, b = model.components["A"], model.components["B"]
        a.trainable = False
        a.inner.trainable = False  # nested layer, reached through .layers
        b.encoder.trainable = False
        before = {
            (n, layer.name): layer.trainable
            for n, comp in model.components.items()
            for layer in comp.all_layers()
        }
        s.prior_initializer.initialize("A")
        after = {
            (n, layer.name): layer.trainable
            for n, comp in model.components.items()
            for layer in comp.all_layers()
        }
        assert after == before
        assert a.trainable is False and b.trainable is True

    def test_restores_state_when_prediction_fails(self, fakes):
        s, model = make_scheduler([make_config("A")])
        model.components["A"].set_progressive_scaler_iteration(2.0, 8.0)

        def broken_predict(*args, **kwargs):
            raise RuntimeError("boom")

        model.predict = broken_predict
        with pytest.raises(RuntimeError):
            s.prior_initializer.initialize("A")
        assert model.trainable_history[-1] is True
        assert model.components["A"].progressive == (2.0, 8.0)

    def test_small_std_is_inverted_exactly(self, fakes):
        s, model, z, _ = self._run(0.5)
        got, expected = self._prior_std_vs_cluster_std(model, z)
        np.testing.assert_allclose(got, expected, rtol=1e-3)

    def test_large_std_is_inverted_exactly(self, fakes):
        # regression: for cluster std > 1 the inverse softplus was
        # approximated by std - 0.5, so the prior std started ~15% below
        # the cluster spread at std 3
        s, model, z, _ = self._run(3.0)
        got, expected = self._prior_std_vs_cluster_std(model, z)
        # softplus(bias) should reproduce the per-cluster std (within 5%)
        np.testing.assert_allclose(got, expected, rtol=0.05)


# ---------------------------------------------------------------------------
# _compile_model
# ---------------------------------------------------------------------------


class TestCompileModel:
    def test_compile_arguments(self, fakes):
        configs = [
            make_config("A", n_vars={"rna": 10, "atac": 30}),
            make_config("B", z=["A"]),
        ]
        s, model = make_scheduler(configs, optimizer="adam", learning_rate=1e-3)
        model._standard_kl_weights["A"].scale = FakeVar(2.0)
        model._gmm_kl_weights["B"].scale = FakeVar(0.5)
        model._data_loss_weights["A"]["atac"].scale = FakeVar(3.0)
        del model._standard_kl_weights["B"]
        s._compile_model()
        kw = model.compile_calls[-1]
        assert kw["standard_kl_weights"] == {"A": 2.0, "B": STANDARD_KL_WEIGHT}
        assert kw["gmm_kl_weights"] == {"A": 1.0, "B": 0.5}
        assert kw["loss_weights"] == {
            f"A_rna_{Constants.MODEL_LOSS_DATA_POSTFIX}": 1.0,
            f"A_atac_{Constants.MODEL_LOSS_DATA_POSTFIX}": 3.0,
            f"B_rna_{Constants.MODEL_LOSS_DATA_POSTFIX}": 1.0,
        }
        opt = kw["optimizer"]
        assert isinstance(opt, tf.keras.optimizers.Adam)
        assert float(opt.learning_rate) == pytest.approx(1e-3)

    def test_optimizer_reused_only_for_same_trainable_set(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs)
        s.state.train_only(["A"])
        s._compile_model()
        opt1 = model.optimizer
        s._compile_model()
        assert model.optimizer is opt1  # same variables -> momentum kept
        s.state.freeze_encoder_path("A")
        s._compile_model()
        opt2 = model.optimizer
        assert opt2 is not opt1
        s.state.train_only(["B"])
        s._compile_model()
        assert model.optimizer is not opt2

    def test_optimizer_name_selects_class(self, fakes):
        s, model = make_scheduler(
            [make_config("A")], optimizer="sgd", learning_rate=0.5
        )
        s._compile_model()
        assert isinstance(model.optimizer, tf.keras.optimizers.SGD)
        assert float(model.optimizer.learning_rate) == pytest.approx(0.5)

    def test_progressive_state_restored_after_compile(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs)
        std = model._standard_kl_weights["A"]
        data = model._data_loss_weights["A"]["rna"]
        std.activate(12)
        std.current_iteration.assign(4.0)
        data.pin_to(0.25)

        def compile_resets(**kwargs):
            model.compile_calls.append(kwargs)
            model.optimizer = kwargs["optimizer"]
            for w in (std, data, model._gmm_kl_weights["A"]):
                w.current_iteration.assign(1.0)
                w.total_iterations.assign(1.0)

        model.compile = compile_resets
        s._compile_model()
        assert (float(std.current_iteration), float(std.total_iterations)) == (
            4.0,
            12.0,
        )
        assert (float(data.current_iteration), float(data.total_iterations)) == (
            0.25,
            1.0,
        )

    def test_z_hat_density_variable_keeps_its_value(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs, learn_z_hat=("A",))
        model._z_hat_density_weights["A"].assign(0.0)
        s._compile_model()
        assert float(model._z_hat_density_weights["A"]) == 0.0


# ---------------------------------------------------------------------------
# mlflow
# ---------------------------------------------------------------------------


class TestMlflow:
    def test_experiment_named_after_model(self, fakes):
        s, model = make_scheduler([make_config("A")])
        exp = s._mlflow_experiment()
        fakes.mlflow.set_experiment.assert_called_once_with("fake_model")
        fakes.mlflow.get_experiment_by_name.assert_called_once_with("fake_model")
        assert exp.experiment_id == "exp-1"

    def test_run_enables_autolog_and_ends(self, fakes):
        s, _ = make_scheduler([make_config("A")])
        exp = fakes.mlflow.get_experiment_by_name.return_value
        with s._mlflow_run("Training/0/A", exp):
            fakes.mlflow.end_run.assert_not_called()
        fakes.mlflow.start_run.assert_called_once_with(
            experiment_id="exp-1", run_name="Training/0/A"
        )
        fakes.mlflow.tensorflow.autolog.assert_called_once_with(
            log_every_epoch=True, log_models=False, checkpoint=False
        )
        fakes.mlflow.end_run.assert_called_once_with()

    def test_run_ends_when_the_body_raises(self, fakes):
        s, _ = make_scheduler([make_config("A")])
        exp = fakes.mlflow.get_experiment_by_name.return_value
        with pytest.raises(RuntimeError):
            with s._mlflow_run("Training/0/A", exp):
                raise RuntimeError("boom")
        fakes.mlflow.end_run.assert_called_once_with()
