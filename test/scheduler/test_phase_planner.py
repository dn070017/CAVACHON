"""PhasePlanner (which phases a component is trained with, and what
each phase sets up) and PhaseProgress (epoch book-keeping)."""

import warnings
from fractions import Fraction

import numpy as np
import pytest

from cavachon.environment.constants import Constants
from cavachon.scheduler.component_schedule import ComponentSchedule
from cavachon.scheduler.phase_planner import PhasePlanner
from cavachon.scheduler.training_phase import PhaseProgress
from test.scheduler.conftest import make_config, make_scheduler


def plan_of(s, model, name, n_batches=3):
    config = next(c for c in model.component_configs if c.name == name)
    order = s.component_order.index(name)
    schedule = ComponentSchedule.from_config(config, order, model, default_epochs=9)
    return s.planner.plan(schedule, n_batches)


def names(phases):
    return [p.name for p in phases]


# ---------------------------------------------------------------------------
# which phases run
# ---------------------------------------------------------------------------


class TestPlan:
    def test_root_with_regular_training_only(self, fakes):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=4)])
        (phase,) = plan_of(s, model, "A")
        assert phase.name == "Regular Training"
        assert phase.run_name == "Training/0/A"
        assert phase.epochs == 4
        assert phase.reset_before is True
        assert phase.early_stopping is True
        assert phase.labels == [("A", 0)] and phase.loss_prefixes == {"A": 0}

    def test_every_phase_in_order(self, fakes):
        configs = [
            make_config("A"),
            make_config(
                "B",
                z=["A"],
                n_parent_annealing_epochs=2,
                n_kl_annealing_epochs=4,
                max_regular_training_epochs=5,
                n_z_hat_annealing_epochs=3,
                n_z_hat_training_epochs=6,
            ),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        phases = plan_of(s, model, "B")
        assert names(phases) == [
            "Parent Annealing",
            "KL Annealing",
            "Regular Training",
            "Z_hat Annealing",
            "Z_hat Training",
        ]
        assert [p.epochs for p in phases] == [2, 4, 5, 3, 6]
        assert [p.run_name for p in phases] == [
            "Training/1/ParentAnnealing/A_to_B",
            "Training/1/KLAnnealing/B",
            "Training/1/B",
            "Z_hat Annealing/1/B",
            "Z_hat Training/1/B",
        ]
        # train_only + recompile before the first three, not the z_hat ones
        assert [p.reset_before for p in phases] == [True, True, True, False, False]
        assert [p.early_stopping for p in phases] == [
            False,
            False,
            True,
            False,
            True,
        ]
        # z_hat training has no AnnealingCallback
        assert phases[-1].schedule is None

    def test_regular_epochs_fall_back_to_fit_epochs(self, fakes):
        s, model = make_scheduler([make_config("A", max_regular_training_epochs=None)])
        assert plan_of(s, model, "A")[0].epochs == 9

    def test_z_hat_training_needs_epochs(self, fakes):
        configs = [
            make_config("A"),
            make_config("B", z_hat=["A"], n_z_hat_annealing_epochs=2),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        assert names(plan_of(s, model, "B"))[-1] == "Z_hat Annealing"


# ---------------------------------------------------------------------------
# k-means epochs
# ---------------------------------------------------------------------------


class TestKMeansEpochs:
    def test_in_kl_phase_at_end_of_standard_kl_stage(self, fakes):
        config = make_config(
            "A",
            n_kl_annealing_epochs=10,
            enable_kmeans_init=True,
            kl_annealing_ratio=(0.4, 0.2, 0.4),
        )
        s, model = make_scheduler([config], learn_z_hat=("A",))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # z_hat prior never trained
            kl, regular = plan_of(s, model, "A")
        assert kl.kmeans_epoch == 4
        # z_hat prior initialized together with z (no z_hat annealing)
        assert kl.z_hat_kmeans_epoch == 4
        assert regular.kmeans_epoch is None and regular.z_hat_kmeans_epoch is None

    def test_in_regular_phase_without_kl_annealing(self, fakes):
        s, model = make_scheduler([make_config("A", enable_kmeans_init=True)])
        with pytest.warns(UserWarning, match="KL annealing is disabled"):
            (regular,) = plan_of(s, model, "A")
        assert regular.kmeans_epoch == 0
        assert regular.z_hat_kmeans_epoch is None  # no z_hat prior

    def test_z_hat_prior_waits_for_z_hat_annealing(self, fakes):
        configs = [
            make_config("A"),
            make_config(
                "B",
                z_hat=["A"],
                n_kl_annealing_epochs=4,
                enable_kmeans_init=True,
                n_z_hat_annealing_epochs=2,
            ),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        kl, regular, _ = plan_of(s, model, "B")
        assert kl.kmeans_epoch == 2
        assert kl.z_hat_kmeans_epoch is None
        assert regular.z_hat_kmeans_epoch is None

    def test_deferred_to_regular_phase_when_standard_kl_fills_kl_phase(self, fakes):
        config = make_config(
            "A",
            n_kl_annealing_epochs=4,
            enable_kmeans_init=True,
            kl_annealing_ratio=(1.0, 0.0, 0.0),
        )
        s, model = make_scheduler([config], learn_z_hat=("A",))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kl, regular = plan_of(s, model, "A")
        # the epoch after the standard-KL stage is the regular phase's first
        assert kl.kmeans_epoch is None and kl.z_hat_kmeans_epoch is None
        assert regular.kmeans_epoch == 0 and regular.z_hat_kmeans_epoch == 0
        assert not any("KL annealing is disabled" in str(w.message) for w in caught)

    def test_disabled(self, fakes):
        s, model = make_scheduler([make_config("A", n_kl_annealing_epochs=4)])
        kl, regular = plan_of(s, model, "A")
        assert kl.kmeans_epoch is None and regular.kmeans_epoch is None


# ---------------------------------------------------------------------------
# what the phases set up
# ---------------------------------------------------------------------------


class TestParentAnnealingHooks:
    def test_setup_and_teardown(self, fakes):
        configs = [
            make_config("A"),
            make_config("B", z=["A"], n_parent_annealing_epochs=2),
        ]
        s, model = make_scheduler(configs)
        parent, _ = plan_of(s, model, "B", n_batches=3)
        assert parent.labels == [("A", 0), ("B", 1)]
        assert parent.loss_prefixes == {"A": 0, "B": 1}
        assert parent.schedule(0) == {}

        parent.setup()
        b = model.snapshot()["B"]
        assert b["progressive"] == (0.0, 6.0)  # alpha 0 -> 1 over 2 * 3 batches
        assert b["standard_kl"] == ("active", 6.0)
        assert b["gmm_kl"] == ("pin", 0.0)
        assert b["data"] == {"rna": ("pin", 1.0)}

        parent.teardown()
        state = model.snapshot()
        assert state["B"]["progressive"] == (1.0, 1.0)
        assert state["B"]["standard_kl"] == ("pin", 1.0)
        assert state["A"]["data"] == {"rna": ("pin", 0.0)}
        assert state["A"]["gmm_kl"] == ("pin", 0.0)


class TestZHatHooks:
    def _setup(self, fakes):
        configs = [
            make_config("A"),
            make_config(
                "B",
                z_hat=["A"],
                n_z_hat_annealing_epochs=4,
                n_z_hat_training_epochs=3,
                z_hat_kl_weight=2.0,
            ),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        rng = np.random.default_rng(0)
        z = np.concatenate(
            [rng.normal(-5, 1, (50, 2)), rng.normal(5, 1, (50, 2))]
        ).astype(np.float32)
        model.predict_outputs = {"B_z_hat": z}
        _, annealing, training = plan_of(s, model, "B")
        return s, model, z, annealing, training

    def test_annealing_setup(self, fakes):
        s, model, z, annealing, _ = self._setup(fakes)
        s.state.train_only(["B"])
        n_compiles = len(model.compile_calls)
        annealing.setup()
        comp = model.components["B"]
        # z_hat prior initialized with k-means (from B_z_hat, not B_z),
        # z_hat std from its spread
        prior = comp.z_hat_prior_parameterizer
        assert all(np.abs(v.numpy()).max() > 2 for v in prior.loc_bias)
        assert np.exp(prior.logits_bias.numpy()).sum() == pytest.approx(1.0)
        assert all(np.all(v.numpy() == 0) for v in comp.z_prior_parameterizer.loc_bias)
        ((args, _),) = comp.hierarchical_encoder.initialize_scale_diag.call_args_list
        np.testing.assert_allclose(args[0], 0.1 * np.std(z, axis=0), rtol=1e-6)
        state = model.snapshot()["B"]
        assert state["layers"]["B_encoder"] is False
        assert state["layers"]["B_decoder"] is True
        assert state["stochastic"] == 1.0
        assert state["z_hat_gmm_kl"] == ("pin", 0.0)
        assert len(model.compile_calls) == n_compiles + 1  # recompiled

    def test_annealing_ramp_and_training_weight(self, fakes):
        s, model, _, annealing, training = self._setup(fakes)
        name = f"B_{Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX}"
        assert [annealing.schedule(e)[name] for e in range(5)] == [
            0.5,
            1.0,
            1.5,
            2.0,
            2.0,
        ]
        training.setup()
        assert model.snapshot()["B"]["z_hat_gmm_kl"] == ("pin", 2.0)


# ---------------------------------------------------------------------------
# PhaseProgress
# ---------------------------------------------------------------------------


class TestPhaseProgress:
    def test_advance_with_full_and_early_stopped_phases(self):
        progress = PhaseProgress(total=14)
        progress.advance(4, 4)
        assert (progress.offset, progress.total, progress.phase_number) == (4, 14, 2)
        progress.advance(5, 2)  # early-stopped after 2 of 5 epochs
        assert (progress.offset, progress.total, progress.phase_number) == (6, 11, 3)


# ---------------------------------------------------------------------------
# epoch boundaries
# ---------------------------------------------------------------------------


class TestEpochBoundary:
    def test_float_error_does_not_move_the_boundary(self):
        assert PhasePlanner.epoch_boundary(10, 0.1 + 0.7) == 8
        assert PhasePlanner.epoch_boundary(10, 0.25) == 2  # floor, not round
        assert PhasePlanner.epoch_boundary(3, 0.5) == 1

    def test_matches_exact_arithmetic_for_all_tenths(self):
        # regression: 12 of 144 ratio/epoch combinations were off by one
        for n in range(1, 17):
            for a in range(11):
                for b in range(11 - a):
                    exact = (n * Fraction(a + b, 10)).__floor__()
                    assert PhasePlanner.epoch_boundary(n, a / 10 + b / 10) == exact
