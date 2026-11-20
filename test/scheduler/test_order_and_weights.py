"""Training order, modality weights, ComponentSchedule and the
TrainingStateController (loss weights, trainable flags, alpha)."""

import pytest

from cavachon.scheduler.component_schedule import (
    DEFAULT_KL_ANNEALING_RATIO,
    ComponentSchedule,
)
from test.scheduler.conftest import FakeVar, make_config, make_scheduler


# ---------------------------------------------------------------------------
# compute_component_training_order
# ---------------------------------------------------------------------------


class TestTrainingOrder:
    def test_chain_is_flattened_in_dependency_order(self, fakes):
        configs = [
            make_config("A"),
            make_config("B", z=["A"]),
            make_config("C", z_hat=["B"]),
        ]
        scheduler, _ = make_scheduler(configs)
        assert scheduler.training_order == [["A"], ["B"], ["C"]]
        assert scheduler.component_order == ["A", "B", "C"]

    def test_siblings_are_trained_one_at_a_time(self, fakes):
        configs = [
            make_config("A"),
            make_config("B"),
            make_config("C", z=["A"], z_hat=["B"]),
            make_config("D", z_hat=["A"]),
        ]
        scheduler, _ = make_scheduler(configs)
        # level 0: A, B; level 1: C, D -> flattened one component per stage
        assert scheduler.training_order == [["A"], ["B"], ["C"], ["D"]]

    def test_level_is_max_parent_level_plus_one(self, fakes):
        configs = [
            make_config("A"),
            make_config("X"),
            make_config("B", z=["A"]),
            make_config("C", z=["B", "X"]),
        ]
        scheduler, _ = make_scheduler(configs)
        # A, X at level 0, B at 1, C at 2 -> X precedes B even though
        # it's listed after A only
        assert scheduler.training_order == [["A"], ["X"], ["B"], ["C"]]

    def test_child_listed_before_parent_is_sorted(self, fakes):
        # regression: the order was not a topological sort, so a child
        # listed before its parent raised KeyError
        configs = [
            make_config("C", z=["B"]),
            make_config("B", z=["A"]),
            make_config("A"),
        ]
        scheduler, model = make_scheduler(configs)
        assert scheduler.training_order == [["A"], ["B"], ["C"]]
        fakes.GeneralUtils.order_components.assert_called_once_with(
            model.component_configs
        )
        assert [c.name for c in scheduler.component_configs] == ["A", "B", "C"]

    @pytest.mark.parametrize(
        "configs, error",
        [
            ([make_config("A"), make_config("B", z=["missing"])], ValueError),
            (
                [make_config("A", z=["B"]), make_config("B", z_hat=["A"])],
                AttributeError,
            ),
        ],
        ids=["unknown parent", "cycle"],
    )
    def test_ordering_errors_propagate(self, fakes, configs, error):
        with pytest.raises(error):
            make_scheduler(configs)


# ---------------------------------------------------------------------------
# compute_modality_weight
# ---------------------------------------------------------------------------


class TestModalityWeight:
    def test_inverse_n_vars_normalised_to_one(self, fakes):
        configs = [make_config("A", n_vars={"rna": 100, "atac": 300})]
        scheduler, _ = make_scheduler(configs)
        w = scheduler.modality_weight["A"]
        # raw weights total/n = 4 and 4/3 -> normalised 0.75 / 0.25
        assert w["rna"] == pytest.approx(0.75)
        assert w["atac"] == pytest.approx(0.25)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_single_modality_weight_is_one(self, fakes):
        scheduler, _ = make_scheduler([make_config("A", n_vars={"rna": 7})])
        assert scheduler.modality_weight["A"] == {"rna": pytest.approx(1.0)}

    def test_per_component(self, fakes):
        configs = [
            make_config("A", n_vars={"rna": 10}),
            make_config("B", z=["A"], n_vars={"rna": 10, "atac": 10}),
        ]
        scheduler, _ = make_scheduler(configs)
        assert scheduler.modality_weight["A"] == {"rna": pytest.approx(1.0)}
        assert scheduler.modality_weight["B"] == {
            "rna": pytest.approx(0.5),
            "atac": pytest.approx(0.5),
        }

    def test_constant_weights_are_one(self, fakes):
        # regression: constant=True read n_vars before assignment
        configs = [
            make_config("A", n_vars={"rna": 100, "atac": 300}),
            make_config("B", z=["A"], n_vars={"atac": 5}),
        ]
        scheduler, _ = make_scheduler(configs)
        w = scheduler.compute_modality_weight(constant=True)
        assert w == {"A": {"rna": 1.0, "atac": 1.0}, "B": {"atac": 1.0}}


# ---------------------------------------------------------------------------
# ComponentSchedule
# ---------------------------------------------------------------------------


class TestComponentSchedule:
    def test_values_read_from_component_config(self, fakes):
        config = make_config(
            "A",
            z=["P"],
            z_hat=["Q"],
            n_parent_annealing_epochs=3,
            n_kl_annealing_epochs=7,
            enable_kmeans_init=True,
            max_regular_training_epochs=11,
            kl_annealing_ratio=(0.2, 0.3, 0.5),
            n_z_hat_annealing_epochs=4,
            n_z_hat_training_epochs=6,
        )
        _, model = make_scheduler([make_config("P"), make_config("Q"), config])
        s = ComponentSchedule.from_config(config, 2, model, default_epochs=99)
        assert s.name == "A" and s.order == 2
        assert s.parents == ["P", "Q"]  # z parents, then z_hat parents
        assert s.n_parent_annealing_epochs == 3
        assert s.n_kl_annealing_epochs == 7
        assert s.enable_kmeans_init is True
        assert s.n_regular_training_epochs == 11
        assert s.kl_annealing_ratios == (0.2, 0.3, 0.5)
        assert s.n_z_hat_annealing_epochs == 4
        assert s.n_z_hat_training_epochs == 6

    def test_none_falls_back(self, fakes):
        config = make_config(
            "A",
            max_regular_training_epochs=None,
            kl_annealing_ratio=None,
            n_z_hat_annealing_epochs=None,
            n_z_hat_training_epochs=None,
        )
        _, model = make_scheduler([config])
        s = ComponentSchedule.from_config(config, 0, model, default_epochs=42)
        assert s.n_regular_training_epochs == 42
        assert s.kl_annealing_ratios == DEFAULT_KL_ANNEALING_RATIO == (0.5, 0.2, 0.3)
        assert s.n_z_hat_annealing_epochs == 0
        assert s.n_z_hat_training_epochs == 0

    def test_phase_flags(self, fakes):
        configs = [
            make_config("A", n_parent_annealing_epochs=3, n_kl_annealing_epochs=0),
            make_config(
                "B", z=["A"], n_parent_annealing_epochs=3, n_kl_annealing_epochs=2
            ),
            make_config("C", z=["A"], n_parent_annealing_epochs=0),
        ]
        _, model = make_scheduler(configs)
        a, b, c = (
            ComponentSchedule.from_config(cfg, i, model, 5)
            for i, cfg in enumerate(configs)
        )
        assert a.runs_parent_annealing is False  # root: no parents
        assert b.runs_parent_annealing is True
        assert c.runs_parent_annealing is False  # 0 epochs
        assert a.runs_kl_annealing is False and b.runs_kl_annealing is True

    def test_z_hat_flags_require_all_conditions(self, fakes):
        configs = [
            make_config("A"),
            make_config("B", z_hat=["A"], n_z_hat_annealing_epochs=2),
            make_config("C", z_hat=["A"], n_z_hat_annealing_epochs=0),
            make_config("D", z_hat=["A"], n_z_hat_annealing_epochs=2),
            make_config(
                "E", z_hat=["A"], n_z_hat_annealing_epochs=2, n_z_hat_training_epochs=3
            ),
        ]
        _, model = make_scheduler(configs, learn_z_hat=("B", "C", "E"))

        def schedule(name):
            cfg = next(c for c in configs if c.name == name)
            return ComponentSchedule.from_config(cfg, 0, model, 5)

        b, c, d, e = (schedule(n) for n in "BCDE")
        assert b.has_z_hat_prior and b.has_z_hat_density_loss
        assert b.z_hat_annealing_enabled is True
        assert b.runs_z_hat_training is False  # 0 training epochs
        assert c.has_z_hat_prior and c.z_hat_annealing_enabled is False  # 0 epochs
        assert d.has_z_hat_prior is False  # no learn_z_hat_priors
        assert d.z_hat_annealing_enabled is False
        assert e.runs_z_hat_training is True
        # the z_hat GMM KL must exist in the compiled model
        del model._z_hat_gmm_kl_weights["B"]
        assert schedule("B").z_hat_annealing_enabled is False
        # the density loss is checked separately
        del model._z_hat_density_weights["B"]
        assert schedule("B").has_z_hat_density_loss is False

    def test_scheduler_resolves_in_training_order(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, _ = make_scheduler(configs)
        schedules = s.compute_component_schedules(default_epochs=8)
        assert [(x.name, x.order) for x in schedules] == [("A", 0), ("B", 1)]


# ---------------------------------------------------------------------------
# TrainingStateController
# ---------------------------------------------------------------------------


class TestLossWeights:
    def test_progressive_scalers_are_pinned(self, fakes):
        configs = [make_config("A", n_vars={"rna": 100, "atac": 300})]
        s, model = make_scheduler(configs, learn_z_hat=("A",))
        s.state.set_loss_weights(
            "A",
            data_scale=2.0,
            gmm_kl=0.3,
            standard_kl=0.7,
            z_hat_density=0.4,
            z_hat_gmm_kl=0.9,
        )
        assert model._gmm_kl_weights["A"].state == ("pin", pytest.approx(0.3))
        assert model._standard_kl_weights["A"].state == ("pin", pytest.approx(0.7))
        assert model._z_hat_gmm_kl_weights["A"].state == ("pin", pytest.approx(0.9))
        assert float(model._z_hat_density_weights["A"]) == pytest.approx(0.4)
        # data weight = modality weight * data_scale
        assert model._data_loss_weights["A"]["rna"].state == ("pin", pytest.approx(1.5))
        assert model._data_loss_weights["A"]["atac"].state == (
            "pin",
            pytest.approx(0.5),
        )

    def test_none_leaves_z_hat_weights_untouched(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs, learn_z_hat=("A",))
        s.state.set_loss_weights("A", z_hat_density=None, z_hat_gmm_kl=None)
        assert model._z_hat_gmm_kl_weights["A"].calls == []
        assert float(model._z_hat_density_weights["A"]) == 1.0

    def test_plain_variables_are_assigned(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs)
        model._gmm_kl_weights["A"] = FakeVar(5.0)
        model._standard_kl_weights["A"] = FakeVar(5.0)
        model._data_loss_weights["A"]["rna"] = FakeVar(5.0)
        s.state.set_loss_weights("A", data_scale=0.5, gmm_kl=0.25, standard_kl=0.75)
        assert float(model._gmm_kl_weights["A"]) == 0.25
        assert float(model._standard_kl_weights["A"]) == 0.75
        assert float(model._data_loss_weights["A"]["rna"]) == 0.5

    def test_zero_loss_weights(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs, learn_z_hat=("A", "B"))
        s.state.zero_loss_weights(["A"])
        for scaler in (
            model._gmm_kl_weights["A"],
            model._standard_kl_weights["A"],
            model._z_hat_gmm_kl_weights["A"],
            model._data_loss_weights["A"]["rna"],
        ):
            assert scaler.state == ("pin", 0.0)
        assert float(model._z_hat_density_weights["A"]) == 0.0
        # the other component is untouched
        assert model._gmm_kl_weights["B"].calls == []
        assert model._data_loss_weights["B"]["rna"].calls == []

    def test_standard_kl_ramp_and_pin(self, fakes):
        s, model = make_scheduler([make_config("A")])
        s.state.start_standard_kl_ramp("A", 12)
        assert model._standard_kl_weights["A"].state == ("active", 12.0)
        s.state.pin_standard_kl("A", 1.0)
        assert model._standard_kl_weights["A"].state == ("pin", 1.0)


class TestAlphaAndSamplers:
    def test_alpha_ramp_and_pin(self, fakes):
        s, model = make_scheduler([make_config("A")])
        s.state.start_alpha_ramp("A", 12)
        assert model.components["A"].progressive_calls[-1] == (0.0, 12.0)
        s.state.pin_alpha("A")
        assert model.components["A"].progressive_calls[-1] == (1.0, 1.0)

    def test_samplers_stochastic(self, fakes):
        s, model = make_scheduler([make_config("A"), make_config("B", z=["A"])])
        s.state.set_samplers_stochastic(z=True, z_hat=False)
        for comp in model.components.values():
            assert float(comp.z_sampler.stochastic) == 1.0
            assert float(comp.z_hat_sampler.stochastic) == 0.0
        s.state.set_z_hat_sampler_stochastic("B")
        assert float(model.components["B"].z_hat_sampler.stochastic) == 1.0
        assert float(model.components["A"].z_hat_sampler.stochastic) == 0.0


class TestTrainOnly:
    def test_trained_component_on_others_frozen_and_zeroed(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        model.components["A"].trainable = True
        s.state.train_only(["B"])

        a, b = model.components["A"], model.components["B"]
        assert a.trainable is False
        assert all(layer.trainable is False for layer in a.all_layers())  # incl. nested
        assert b.trainable is True
        assert all(layer.trainable is True for layer in b.all_layers())

        assert model._gmm_kl_weights["A"].state == ("pin", 0.0)
        assert model._data_loss_weights["A"]["rna"].state == ("pin", 0.0)
        assert model._gmm_kl_weights["B"].state == ("pin", 1.0)
        assert model._data_loss_weights["B"]["rna"].state == ("pin", 1.0)
        for n in ("A", "B"):
            assert model._standard_kl_weights[n].state == ("pin", 0.0)
        assert model._z_hat_gmm_kl_weights["B"].state == ("pin", 0.0)
        assert float(model._z_hat_density_weights["B"]) == 0.0
        # progressive scaler is pinned at alpha = 1 for everyone
        assert a.progressive == (1.0, 1.0)
        assert b.progressive == (1.0, 1.0)

    def test_unfreezes_previously_frozen_layers(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs)
        comp = model.components["A"]
        for layer in comp.all_layers():
            layer.trainable = False
        s.state.train_only(["A"])
        assert all(layer.trainable for layer in comp.all_layers())

    def test_freeze_encoder_path(self, fakes):
        s, model = make_scheduler([make_config("A")])
        comp = model.components["A"]
        s.state.freeze_encoder_path("A")
        assert comp.preprocessor.trainable is False
        assert comp.encoder.trainable is False
        assert comp.z_prior_parameterizer.trainable is False
        assert comp.decoder.trainable is True
        assert comp.inner.trainable is True
