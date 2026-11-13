"""Phase orchestration of SequentialTrainingScheduler.fit: which phases run,
what is trained/frozen in each, the annealing values, the fit arguments,
the callbacks passed and what is logged to mlflow."""

import warnings

import numpy as np
import pytest

from cavachon.environment.constants import Constants
from test.scheduler.conftest import UserCallback, callbacks_of, make_config, make_scheduler


def run_names(fakes):
    return [c.kwargs["run_name"] for c in fakes.mlflow.start_run.call_args_list]


def phases(model, fakes):
    return [
        callbacks_of(call, fakes.VerboseCallback)[0].kwargs["phase"]
        for call in model.fit_calls
    ]


# ---------------------------------------------------------------------------
# single root component
# ---------------------------------------------------------------------------


class TestSingleRoot:
    def test_regular_phase_only(self, fakes, dataset):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=5)])
        history = s.fit(dataset, epochs=400, shuffle=False)

        assert len(history) == 1 and history[0].epoch == list(range(5))
        (call,) = model.fit_calls
        assert call.epochs == 5
        assert call.verbose == 0
        assert call.kwargs == {"shuffle": False}  # 'epochs' stripped
        types = [type(cb).__name__ for cb in call.callbacks]
        assert types == [
            "VerboseCallback", "AnnealingCallback",
            "OptimizerStateCallback", "EarlyStoppingCallback",
        ]
        ann = callbacks_of(call, fakes.AnnealingCallback)[0].kwargs
        assert ann["kmeans_epoch"] is None
        assert ann["schedule"](0) == {}
        assert ann["scheduler"] is s and ann["component_name"] == "A"

        state = call.state["A"]
        assert state["trainable"] is True
        assert state["gmm_kl"] == ("pin", 1.0)
        assert state["standard_kl"] == ("pin", 0.0)
        assert state["data"] == {"rna": ("pin", 1.0)}
        assert state["progressive"] == (1.0, 1.0)

    def test_mlflow_runs(self, fakes, dataset):
        s, _ = make_scheduler([make_config("A", max_regular_training_epochs=2)])
        s.fit(dataset, epochs=2)
        fakes.mlflow.set_experiment.assert_called_once_with("fake_model")
        assert run_names(fakes) == ["Training/0/A"]
        assert fakes.mlflow.start_run.call_args.kwargs["experiment_id"] == "exp-1"
        assert fakes.mlflow.end_run.call_count == 1

    def test_early_stopping_arguments(self, fakes, dataset):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=2)])
        s.fit(dataset, epochs=400)
        es = callbacks_of(model.fit_calls[0], fakes.EarlyStoppingCallback)[0].kwargs
        assert es == dict(
            cumulative_offset=0, monitor="loss", min_delta=5, patience=20,
            restore_best_weights=True, verbose=0,
        )

    def test_early_stopping_patience_floor(self, fakes, dataset):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=2)])
        s.fit(dataset, epochs=40)
        es = callbacks_of(model.fit_calls[0], fakes.EarlyStoppingCallback)[0]
        assert es.kwargs["patience"] == 10

    def test_early_stopping_disabled(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", max_regular_training_epochs=2)], early_stopping=False
        )
        s.fit(dataset, epochs=2)
        assert callbacks_of(model.fit_calls[0], fakes.EarlyStoppingCallback) == []

    def test_periodic_tsne_only_with_outdir(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", max_regular_training_epochs=2)],
            outdir="/out", batch_size=64,
            distribution_names={"rna": "NB"}, batch_effect_colnames={"rna": ["b"]},
        )
        s.fit(dataset, epochs=2)
        (tsne,) = callbacks_of(model.fit_calls[0], fakes.PeriodicTSNECallback)
        assert tsne.kwargs == dict(
            mdata="MDATA", component="A", outdir="/out/snapshots", batch_size=64,
            batch_effect_colnames={"rna": ["b"]}, distribution_names={"rna": "NB"},
        )

    def test_no_periodic_tsne_for_multiple_components(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", max_regular_training_epochs=1),
             make_config("B", z=["A"], max_regular_training_epochs=1)],
            outdir="/out",
        )
        s.fit(dataset, epochs=2)
        assert all(
            callbacks_of(c, fakes.PeriodicTSNECallback) == [] for c in model.fit_calls
        )

    def test_regular_epochs_fall_back_to_fit_epochs(self, fakes, dataset):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=None)])
        s.fit(dataset, epochs=7)
        assert model.fit_calls[0].epochs == 7

    def test_kmeans_without_kl_annealing_runs_in_regular_phase(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", enable_kmeans_init=True, max_regular_training_epochs=2)],
            learn_z_hat=("A",),
        )
        with pytest.warns(UserWarning, match="KL annealing is disabled"):
            s.fit(dataset, epochs=2)
        anns = callbacks_of(model.fit_calls[0], fakes.AnnealingCallback)
        assert [a.kwargs["kmeans_epoch"] for a in anns] == [0, 0]
        assert "kmeans_kwargs" not in anns[0].kwargs
        assert anns[1].kwargs["kmeans_kwargs"] == {
            "representation_key": "_z_hat",
            "target_parameterizer_attr": "z_hat_prior_parameterizer",
        }

    def test_forwards_other_fit_kwargs_to_each_phase(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", n_kl_annealing_epochs=2, max_regular_training_epochs=2)]
        )
        s.fit(dataset, epochs=2, shuffle=True, class_weight=None)
        assert [c.kwargs for c in model.fit_calls] == [
            {"shuffle": True, "class_weight": None}
        ] * 2

    def test_user_callbacks_are_copied_into_every_phase(self, fakes, dataset):
        # regression: fixed bug, see UNIT_TEST_REPORT.md
        user_cb = UserCallback("mine")
        s, model = make_scheduler(
            [make_config("A", n_kl_annealing_epochs=2, max_regular_training_epochs=2)]
        )
        s.fit(dataset, epochs=2, callbacks=[user_cb])
        assert len(model.fit_calls) == 2
        for call in model.fit_calls:
            first = call.callbacks[0]
            assert isinstance(first, UserCallback) and first.tag == "mine"
            assert first is not user_cb

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: mlflow.end_run is not called when model.fit raises, leaving "
        "an active run that breaks the next start_run",
    )
    def test_mlflow_run_closed_when_fit_raises(self, fakes, dataset):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=2)])
        model.fit_side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            s.fit(dataset, epochs=2)
        assert fakes.mlflow.end_run.call_count == fakes.mlflow.start_run.call_count


# ---------------------------------------------------------------------------
# KL annealing
# ---------------------------------------------------------------------------


class TestKLAnnealing:
    def _fit(self, fakes, dataset, learn_z_hat=(), **cfg):
        cfg.setdefault("n_kl_annealing_epochs", 10)
        cfg.setdefault("max_regular_training_epochs", 5)
        s, model = make_scheduler([make_config("A", **cfg)], learn_z_hat=learn_z_hat)
        s.fit(dataset, epochs=5)
        return s, model

    def test_phase_order_and_runs(self, fakes, dataset):
        s, model = self._fit(fakes, dataset, kl_annealing_ratio=(0.5, 0.2, 0.3))
        assert phases(model, fakes) == ["KL Annealing", "Regular Training"]
        assert [c.epochs for c in model.fit_calls] == [10, 5]
        assert run_names(fakes) == ["Training/0/KLAnnealing/A", "Training/0/A"]
        assert fakes.mlflow.end_run.call_count == 2

    def test_kl_phase_callbacks_and_state(self, fakes, dataset):
        s, model = self._fit(
            fakes, dataset, kl_annealing_ratio=(0.4, 0.2, 0.4), enable_kmeans_init=True,
        )
        kl = model.fit_calls[0]
        assert [type(cb).__name__ for cb in kl.callbacks] == [
            "VerboseCallback", "AnnealingCallback", "OptimizerStateCallback",
        ]  # no early stopping while the loss changes by design
        ann = callbacks_of(kl, fakes.AnnealingCallback)[0].kwargs
        assert ann["kmeans_epoch"] == 4  # end of the standard-KL-only stage
        std = f"A_{Constants.MODEL_LOSS_STANDARD_KL_POSTFIX}"
        gmm = f"A_{Constants.MODEL_LOSS_GMM_KL_POSTFIX}"
        assert ann["schedule"](0)[std] == 1.0 and ann["schedule"](0)[gmm] == 0.0
        assert ann["schedule"](9)[std] == 0.0 and ann["schedule"](9)[gmm] == 1.0
        assert kl.state["A"]["gmm_kl"] == ("pin", 1.0)
        assert kl.state["A"]["data"] == {"rna": ("pin", 1.0)}
        # k-means already done -> not repeated in the regular phase
        reg = callbacks_of(model.fit_calls[1], fakes.AnnealingCallback)
        assert [a.kwargs["kmeans_epoch"] for a in reg] == [None]

    def test_kmeans_disabled(self, fakes, dataset):
        s, model = self._fit(fakes, dataset, enable_kmeans_init=False)
        anns = callbacks_of(model.fit_calls[0], fakes.AnnealingCallback)
        assert [a.kwargs["kmeans_epoch"] for a in anns] == [None]

    def test_z_hat_kmeans_in_kl_phase(self, fakes, dataset):
        s, model = self._fit(
            fakes, dataset, learn_z_hat=("A",), enable_kmeans_init=True,
            kl_annealing_ratio=(0.5, 0.2, 0.3),
        )
        anns = callbacks_of(model.fit_calls[0], fakes.AnnealingCallback)
        assert len(anns) == 2
        assert anns[1].kwargs["kmeans_epoch"] == 5
        assert anns[1].kwargs["kmeans_kwargs"]["representation_key"] == "_z_hat"
        assert anns[1].kwargs["schedule"](3) == {}

    def test_warns_when_kmeans_epoch_is_zero(self, fakes, dataset):
        with pytest.warns(UserWarning, match="Kmeans initialization epoch is 0"):
            self._fit(
                fakes, dataset, n_kl_annealing_epochs=1, enable_kmeans_init=True,
                kl_annealing_ratio=(0.5, 0.2, 0.3),
            )

    def test_no_warning_with_kl_annealing(self, fakes, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._fit(fakes, dataset, enable_kmeans_init=True,
                      kl_annealing_ratio=(0.5, 0.2, 0.3))

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: kl_annealing_ratio[0] == 1 puts kmeans_epoch == n_epochs (never "
        "reached) yet the regular phase skips k-means, so it never runs",
    )
    def test_kmeans_fires_somewhere_with_full_standard_kl_stage(self, fakes, dataset):
        s, model = self._fit(
            fakes, dataset, n_kl_annealing_epochs=4, enable_kmeans_init=True,
            kl_annealing_ratio=(1.0, 0.0, 0.0),
        )
        fired = [
            a.kwargs["kmeans_epoch"]
            for call in model.fit_calls
            for a in callbacks_of(call, fakes.AnnealingCallback)
            if a.kwargs["kmeans_epoch"] is not None
            and 0 <= a.kwargs["kmeans_epoch"] < call.epochs
        ]
        assert fired


# ---------------------------------------------------------------------------
# parent annealing (commit 73eecd0)
# ---------------------------------------------------------------------------


def _parent_child(**child_cfg):
    child_cfg.setdefault("n_parent_annealing_epochs", 3)
    child_cfg.setdefault("max_regular_training_epochs", 4)
    return [
        make_config("A", max_regular_training_epochs=2),
        make_config("B", z=["A"], **child_cfg),
    ]


class TestParentAnnealing:
    def test_phase_order(self, fakes, dataset):
        s, model = make_scheduler(_parent_child())
        history = s.fit(dataset, epochs=5)
        assert phases(model, fakes) == [
            "Regular Training", "Parent Annealing", "Regular Training",
        ]
        assert [c.epochs for c in model.fit_calls] == [2, 3, 4]
        assert len(history) == 3
        assert run_names(fakes) == [
            "Training/0/A", "Training/1/ParentAnnealing/A_to_B", "Training/1/B",
        ]

    def test_parents_frozen_and_zeroed(self, fakes, dataset):
        s, model = make_scheduler(_parent_child())
        s.fit(dataset, epochs=5)
        a = model.fit_calls[1].state["A"]
        assert a["trainable"] is False
        assert not any(a["layers"].values())
        assert a["gmm_kl"] == ("pin", 0.0)
        assert a["standard_kl"] == ("pin", 0.0)
        assert a["data"] == {"rna": ("pin", 0.0)}
        # the parent's own progressive scaler is not ramped
        assert a["progressive"] == (1.0, 1.0)

    def test_child_alpha_and_standard_kl_ramp_together(self, fakes, dataset):
        s, model = make_scheduler(_parent_child())
        s.fit(dataset, epochs=5)
        b = model.fit_calls[1].state["B"]
        n_iter = 3 * len(dataset)  # n_batches * n_prog_epochs
        assert b["trainable"] is True and all(b["layers"].values())
        assert b["progressive"] == (0.0, float(n_iter))  # alpha 0 -> 1
        assert b["standard_kl"] == ("active", float(n_iter))  # tied to alpha
        # plain VAE loss: data weight 1, no GMM KL
        assert b["data"] == {"rna": ("pin", 1.0)}
        assert b["gmm_kl"] == ("pin", 0.0)

    def test_child_z_hat_losses_off(self, fakes, dataset):
        s, model = make_scheduler(_parent_child(), learn_z_hat=("B",))
        s.fit(dataset, epochs=5)
        b = model.fit_calls[1].state["B"]
        assert b["z_hat_gmm_kl"] == ("pin", 0.0)

    def test_callbacks_have_no_kmeans_and_no_early_stopping(self, fakes, dataset):
        s, model = make_scheduler(_parent_child(enable_kmeans_init=True))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.fit(dataset, epochs=5)
        call = model.fit_calls[1]
        assert [type(cb).__name__ for cb in call.callbacks] == [
            "VerboseCallback", "AnnealingCallback", "OptimizerStateCallback",
        ]
        ann = callbacks_of(call, fakes.AnnealingCallback)[0].kwargs
        assert ann.get("kmeans_epoch") is None
        assert ann["schedule"](0) == {} and ann["schedule"](2) == {}
        assert ann["component_name"] == "B"
        verbose = callbacks_of(call, fakes.VerboseCallback)[0].kwargs
        assert verbose["labels"] == [("A", 0), ("B", 1)]
        assert verbose["phase_epochs"] == 3

    def test_after_phase_alpha_pinned_and_kl_pinned_to_one(self, fakes, dataset):
        s, model = make_scheduler(_parent_child())
        s.fit(dataset, epochs=5)
        n_iter = 3.0 * len(dataset)
        b_std = model._standard_kl_weights["B"].calls
        i = b_std.index(("active", n_iter))
        assert b_std[i + 1] == ("pin", 1.0)
        # alpha pinned back to 1 right after the phase
        b_prog = model.components["B"].progressive_calls
        j = b_prog.index((0.0, n_iter))
        assert b_prog[j + 1] == (1.0, 1.0)
        # regular training: alpha 1, standard KL off, GMM KL on
        b = model.fit_calls[2].state["B"]
        assert b["progressive"] == (1.0, 1.0)
        assert b["standard_kl"] == ("pin", 0.0)
        assert b["gmm_kl"] == ("pin", 1.0)

    def test_multiple_parents_z_and_z_hat(self, fakes, dataset):
        configs = [
            make_config("A", max_regular_training_epochs=1),
            make_config("B", max_regular_training_epochs=1),
            make_config("C", z=["A"], z_hat=["B"], n_parent_annealing_epochs=2,
                        max_regular_training_epochs=1),
        ]
        s, model = make_scheduler(configs)
        s.fit(dataset, epochs=1)
        assert "Training/2/ParentAnnealing/A-B_to_C" in run_names(fakes)
        call = model.fit_calls[2]
        for p in ("A", "B"):
            assert call.state[p]["trainable"] is False
            assert call.state[p]["data"] == {"rna": ("pin", 0.0)}
        verbose = callbacks_of(call, fakes.VerboseCallback)[0].kwargs
        assert verbose["labels"] == [("A", 0), ("B", 0), ("C", 1)]

    def test_skipped_when_zero_epochs(self, fakes, dataset):
        s, model = make_scheduler(_parent_child(n_parent_annealing_epochs=0))
        s.fit(dataset, epochs=5)
        assert phases(model, fakes) == ["Regular Training", "Regular Training"]

    def test_root_never_runs_parent_annealing(self, fakes, dataset):
        s, model = make_scheduler(
            [make_config("A", n_parent_annealing_epochs=3, max_regular_training_epochs=1)]
        )
        s.fit(dataset, epochs=1)
        assert phases(model, fakes) == ["Regular Training"]

    def test_with_kl_annealing_after(self, fakes, dataset):
        s, model = make_scheduler(_parent_child(n_kl_annealing_epochs=4))
        s.fit(dataset, epochs=5)
        assert phases(model, fakes) == [
            "Regular Training", "Parent Annealing", "KL Annealing", "Regular Training",
        ]
        kl = model.fit_calls[2].state
        # KL annealing starts with alpha back at 1 and parents still zeroed
        assert kl["B"]["progressive"] == (1.0, 1.0)
        assert kl["A"]["data"] == {"rna": ("pin", 0.0)}
        assert kl["A"]["trainable"] is False


# ---------------------------------------------------------------------------
# z_hat annealing / training
# ---------------------------------------------------------------------------


def _z_hat_setup(fakes, **cfg):
    cfg.setdefault("n_z_hat_annealing_epochs", 4)
    cfg.setdefault("n_z_hat_training_epochs", 3)
    configs = [
        make_config("A", max_regular_training_epochs=2),
        make_config("B", z_hat=["A"], max_regular_training_epochs=2,
                    z_hat_kl_weight=2.0, **cfg),
    ]
    s, model = make_scheduler(configs, learn_z_hat=("B",))
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(-5, 1, (50, 2)), rng.normal(5, 1, (50, 2))]).astype(np.float32)
    model.predict_outputs = {"B_z_hat": z}
    return s, model, z


class TestZHatPhases:
    def test_phase_order_and_runs(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        assert phases(model, fakes) == [
            "Regular Training", "Regular Training", "Z_hat Annealing", "Z_hat Training",
        ]
        assert [c.epochs for c in model.fit_calls] == [2, 2, 4, 3]
        assert run_names(fakes) == [
            "Training/0/A", "Training/1/B", "Z_hat Annealing/1/B", "Z_hat Training/1/B",
        ]
        assert fakes.mlflow.end_run.call_count == 4

    def test_annealing_state(self, fakes, dataset):
        s, model, z = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        call = model.fit_calls[2]
        b = call.state["B"]
        assert b["layers"]["B_preprocessor"] is False
        assert b["layers"]["B_encoder"] is False
        assert b["layers"]["B_z_prior"] is False
        assert b["layers"]["B_decoder"] is True
        assert b["layers"]["B_z_hat_prior"] is True
        assert b["stochastic"] == 1.0
        assert b["gmm_kl"] == ("pin", 1.0)
        assert b["z_hat_gmm_kl"] == ("pin", 0.0)  # ramped by the callback
        assert call.state["A"]["trainable"] is False
        # optimizer rebuilt for the reduced variable set
        assert call.optimizer is not model.fit_calls[1].optimizer

    def test_annealing_initialises_z_hat_prior_and_scale(self, fakes, dataset):
        s, model, z = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        comp = model.components["B"]
        means = sorted(float(v.numpy()[0, 0]) for v in comp.z_hat_prior_parameterizer.loc_bias)
        assert means[0] < -2 and means[1] > 2
        (args, _), = comp.hierarchical_encoder.initialize_scale_diag.call_args_list
        np.testing.assert_allclose(args[0], 0.1 * np.std(z, axis=0), rtol=1e-6)

    def test_annealing_schedule_and_callbacks(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        call = model.fit_calls[2]
        assert [type(cb).__name__ for cb in call.callbacks] == [
            "VerboseCallback", "AnnealingCallback",
        ]
        ann = callbacks_of(call, fakes.AnnealingCallback)[0].kwargs
        name = f"B_{Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX}"
        assert [ann["schedule"](e) for e in range(4)] == [
            {name: 0.0}, {name: 0.5}, {name: 1.0}, {name: 1.5},
        ]
        assert ann["schedule"](10) == {name: 2.0}  # capped at the target

    def test_training_state_and_callbacks(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        call = model.fit_calls[3]
        b = call.state["B"]
        assert b["z_hat_gmm_kl"] == ("pin", 2.0)
        assert b["layers"]["B_encoder"] is False  # still frozen
        assert b["stochastic"] == 1.0
        assert [type(cb).__name__ for cb in call.callbacks] == [
            "VerboseCallback", "OptimizerStateCallback", "EarlyStoppingCallback",
        ]
        # same variables as the annealing phase -> optimizer reused
        assert call.optimizer is model.fit_calls[2].optimizer

    def test_training_skipped_when_zero(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes, n_z_hat_training_epochs=0)
        s.fit(dataset, epochs=5)
        assert phases(model, fakes)[-1] == "Z_hat Annealing"

    def test_disabled_without_annealing_epochs(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes, n_z_hat_annealing_epochs=0)
        s.fit(dataset, epochs=5)
        assert phases(model, fakes) == ["Regular Training", "Regular Training"]

    def test_next_component_unfreezes_nothing_of_previous(self, fakes, dataset):
        configs = [
            make_config("A", max_regular_training_epochs=1),
            make_config("B", z_hat=["A"], max_regular_training_epochs=1,
                        n_z_hat_annealing_epochs=1),
            make_config("C", z_hat=["B"], max_regular_training_epochs=1),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        model.predict_outputs = {"B_z_hat": np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)}
        s.fit(dataset, epochs=1)
        c_call = model.fit_calls[-1]
        assert c_call.state["B"]["trainable"] is False
        assert not any(c_call.state["B"]["layers"].values())
        assert c_call.state["B"]["z_hat_gmm_kl"] == ("pin", 0.0)
        assert all(c_call.state["C"]["layers"].values())


# ---------------------------------------------------------------------------
# epoch book-keeping across phases
# ---------------------------------------------------------------------------


class TestCumulativeEpochs:
    def test_offsets_totals_and_phase_numbers(self, fakes, dataset):
        configs = [
            make_config("A", n_kl_annealing_epochs=4, max_regular_training_epochs=5,
                        kl_annealing_ratio=(0.5, 0.2, 0.3)),
            make_config("B", z=["A"], n_parent_annealing_epochs=2,
                        max_regular_training_epochs=3),
        ]
        s, model = make_scheduler(configs)
        # A's regular phase early-stops after 2 of its 5 epochs
        model.epochs_run = [None, 2, None, None]
        s.fit(dataset, epochs=5)
        verbose = [callbacks_of(c, fakes.VerboseCallback)[0].kwargs for c in model.fit_calls]
        got = [(v["cumulative_offset"], v["cumulative_total"], v["phase_number"]) for v in verbose]
        assert got == [(0, 14, 1), (4, 14, 2), (6, 11, 3), (8, 11, 4)]
        assert verbose[0]["component_order"] == ["A", "B"]
        es = [callbacks_of(c, fakes.EarlyStoppingCallback) for c in model.fit_calls]
        assert [e[0].kwargs["cumulative_offset"] for e in es if e] == [4, 8]

    def test_total_includes_z_hat_phases(self, fakes, dataset):
        s, model, _ = _z_hat_setup(fakes)
        s.fit(dataset, epochs=5)
        verbose = [callbacks_of(c, fakes.VerboseCallback)[0].kwargs for c in model.fit_calls]
        got = [(v["cumulative_offset"], v["cumulative_total"], v["phase_number"]) for v in verbose]
        assert got == [(0, 11, 1), (2, 11, 2), (4, 11, 3), (8, 11, 4)]

    def test_second_fit_starts_from_scratch(self, fakes, dataset):
        configs = _parent_child()
        s, model = make_scheduler(configs)
        s.fit(dataset, epochs=5)
        first = [callbacks_of(c, fakes.VerboseCallback)[0].kwargs for c in model.fit_calls]
        model.fit_calls.clear()
        s.fit(dataset, epochs=5)
        second = [callbacks_of(c, fakes.VerboseCallback)[0].kwargs for c in model.fit_calls]
        key = lambda v: (v["cumulative_offset"], v["cumulative_total"], v["phase_number"])
        assert [key(v) for v in first] == [key(v) for v in second]
        assert model.fit_calls[1].state["B"]["progressive"] == (0.0, 9.0)
