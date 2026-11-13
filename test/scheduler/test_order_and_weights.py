"""Training order, modality weights, config getters and the loss-weight /
trainable-flag helpers of SequentialTrainingScheduler."""

import pytest

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
        assert scheduler.run_progressive_training == {"A": False, "B": True, "C": True}

    def test_siblings_are_trained_one_at_a_time(self, fakes):
        configs = [
            make_config("A"),
            make_config("B"),
            make_config("C", z=["A"], z_hat=["B"]),
            make_config("D", z_hat=["A"]),
        ]
        scheduler, _ = make_scheduler(configs)
        # level 0: A, B; level 1: C, D -> flattened one component per step
        assert scheduler.training_order == [["A"], ["B"], ["C"], ["D"]]
        assert scheduler.run_progressive_training == {
            "A": False, "B": False, "C": True, "D": True,
        }

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

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: training order is not a topological sort; a child listed "
        "before its parent raises KeyError",
    )
    def test_child_listed_before_parent_is_sorted(self, fakes):
        configs = [make_config("B", z=["A"]), make_config("A")]
        scheduler, _ = make_scheduler(configs)
        assert scheduler.training_order == [["A"], ["B"]]


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
            "rna": pytest.approx(0.5), "atac": pytest.approx(0.5),
        }

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: constant=True reads n_vars before assignment (UnboundLocalError)",
    )
    def test_constant_weights_are_one(self, fakes):
        configs = [make_config("A", n_vars={"rna": 100, "atac": 300})]
        scheduler, _ = make_scheduler(configs)
        w = scheduler.compute_modality_weight(constant=True)
        assert w["A"] == {"rna": 1.0, "atac": 1.0}


# ---------------------------------------------------------------------------
# config getters
# ---------------------------------------------------------------------------


class TestConfigGetters:
    def test_values_read_from_component_config(self, fakes):
        configs = [
            make_config(
                "A",
                n_parent_annealing_epochs=3,
                n_kl_annealing_epochs=7,
                enable_kmeans_init=True,
                max_regular_training_epochs=11,
                kl_annealing_ratio=(0.2, 0.3, 0.5),
                n_z_hat_annealing_epochs=4,
                n_z_hat_training_epochs=6,
            )
        ]
        s, _ = make_scheduler(configs)
        assert s._get_parent_annealing_epochs("A") == 3
        assert s._get_kl_annealing_epochs("A") == 7
        assert s._get_enable_kmeans_init("A") is True
        assert s._get_max_regular_training_epochs("A") == 11
        assert s._get_kl_annealing_ratios("A") == (0.2, 0.3, 0.5)
        assert s._get_z_hat_annealing_epochs("A") == 4
        assert s._get_z_hat_training_epochs("A") == 6

    def test_defaults_for_unknown_component(self, fakes):
        s, _ = make_scheduler([make_config("A")])
        assert s._get_parent_annealing_epochs("nope") == 0
        assert s._get_kl_annealing_epochs("nope") == 0
        assert s._get_enable_kmeans_init("nope") is True
        assert s._get_kl_annealing_ratios("nope") == (0.5, 0.2, 0.3)
        assert s._get_z_hat_annealing_epochs("nope") == 0
        assert s._get_z_hat_training_epochs("nope") == 0
        assert s._get_max_regular_training_epochs("nope") == 100

    def test_none_falls_back(self, fakes):
        configs = [
            make_config(
                "A",
                max_regular_training_epochs=None,
                kl_annealing_ratio=None,
                n_z_hat_annealing_epochs=None,
                n_z_hat_training_epochs=None,
            )
        ]
        s, _ = make_scheduler(configs)
        s._fit_epochs = 42
        assert s._get_max_regular_training_epochs("A") == 42
        assert s._get_kl_annealing_ratios("A") == (0.5, 0.2, 0.3)
        assert s._get_z_hat_annealing_epochs("A") == 0
        assert s._get_z_hat_training_epochs("A") == 0

    def test_parent_component_includes_z_and_z_hat(self, fakes):
        configs = [
            make_config("A"),
            make_config("B"),
            make_config("C", z=["A"], z_hat=["B"]),
        ]
        s, _ = make_scheduler(configs)
        assert s._get_parent_component("C") == ["A", "B"]
        assert s._get_parent_component("A") == []
        assert s._get_parent_component("missing") == []

    def test_z_hat_annealing_enabled_requires_all_conditions(self, fakes):
        configs = [
            make_config("A"),
            make_config("B", z_hat=["A"], n_z_hat_annealing_epochs=2),
            make_config("C", z_hat=["A"], n_z_hat_annealing_epochs=0),
            make_config("D", z_hat=["A"], n_z_hat_annealing_epochs=2),
        ]
        s, model = make_scheduler(configs, learn_z_hat=("B", "C"))
        assert s._z_hat_annealing_enabled("B") is True
        assert s._z_hat_annealing_enabled("C") is False  # 0 epochs
        assert s._z_hat_annealing_enabled("D") is False  # no learn_z_hat_priors
        del model._z_hat_gmm_kl_weights["B"]
        assert s._z_hat_annealing_enabled("B") is False


# ---------------------------------------------------------------------------
# _set_component_weights / _zero_component_variables / progressive
# ---------------------------------------------------------------------------


class TestSetComponentWeights:
    def test_progressive_scalers_are_pinned(self, fakes):
        configs = [make_config("A", n_vars={"rna": 100, "atac": 300})]
        s, model = make_scheduler(configs, learn_z_hat=("A",))
        s._set_component_weights(
            "A", data_scale=2.0, gmm_kl=0.3, standard_kl=0.7,
            z_hat_density=0.4, z_hat_gmm_kl=0.9,
        )
        assert model._gmm_kl_weights["A"].state == ("pin", pytest.approx(0.3))
        assert model._standard_kl_weights["A"].state == ("pin", pytest.approx(0.7))
        assert model._z_hat_gmm_kl_weights["A"].state == ("pin", pytest.approx(0.9))
        assert float(model._z_hat_density_weights["A"]) == pytest.approx(0.4)
        # data weight = modality weight * data_scale
        assert model._data_loss_weights["A"]["rna"].state == ("pin", pytest.approx(1.5))
        assert model._data_loss_weights["A"]["atac"].state == ("pin", pytest.approx(0.5))

    def test_none_leaves_z_hat_weights_untouched(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs, learn_z_hat=("A",))
        s._set_component_weights("A", z_hat_density=None, z_hat_gmm_kl=None)
        assert model._z_hat_gmm_kl_weights["A"].calls == []
        assert float(model._z_hat_density_weights["A"]) == 1.0

    def test_plain_variables_are_assigned(self, fakes):
        configs = [make_config("A")]
        s, model = make_scheduler(configs)
        model._gmm_kl_weights["A"] = FakeVar(5.0)
        model._standard_kl_weights["A"] = FakeVar(5.0)
        model._data_loss_weights["A"]["rna"] = FakeVar(5.0)
        s._set_component_weights("A", data_scale=0.5, gmm_kl=0.25, standard_kl=0.75)
        assert float(model._gmm_kl_weights["A"]) == 0.25
        assert float(model._standard_kl_weights["A"]) == 0.75
        assert float(model._data_loss_weights["A"]["rna"]) == 0.5

    def test_zero_component_variables(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs, learn_z_hat=("A", "B"))
        s._zero_component_variables(["A"])
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

    def test_set_component_progressive(self, fakes):
        s, model = make_scheduler([make_config("A")])
        s._set_component_progressive("A", active=True, n_batches=3, n_epochs=4)
        assert model.components["A"].progressive_calls[-1] == (0.0, 12.0)
        s._set_component_progressive("A", active=False)
        assert model.components["A"].progressive_calls[-1] == (1.0, 1.0)


class TestSetupComponentAndLossWeights:
    def test_trained_component_on_others_frozen_and_zeroed(self, fakes):
        configs = [make_config("A"), make_config("B", z=["A"])]
        s, model = make_scheduler(configs, learn_z_hat=("B",))
        model.components["A"].trainable = True
        s.setup_component_and_loss_weights(["B"])

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
        s.setup_component_and_loss_weights(["A"])
        assert all(layer.trainable for layer in comp.all_layers())

    def test_freeze_encoder_path(self, fakes):
        s, model = make_scheduler([make_config("A")])
        comp = model.components["A"]
        s._freeze_encoder_path("A")
        assert comp.preprocessor.trainable is False
        assert comp.encoder.trainable is False
        assert comp.z_prior_parameterizer.trainable is False
        assert comp.decoder.trainable is True
        assert comp.inner.trainable is True
