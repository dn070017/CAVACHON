"""Model.make / setup_* / prepare_* / __setattr__ (cavachon/model/model.py).

Mocked: Component (FakeComponentFactory -> tiny real keras Models),
GeneralUtils.order_components, and ComponentConfig (FakeConfig). All other
model.py dependencies are patched by the autouse ``fakes`` fixture.
"""

import numpy as np
import pytest
import tensorflow as tf

import cavachon.model.model as model_module
from cavachon.environment.constants import Constants
from test.model.conftest import FakeComponent, FakeConfig, make_batch

Model = model_module.Model


def _values(t):
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t)


# ---------------------------------------------------------------------------
# setup_components / order
# ---------------------------------------------------------------------------


class TestSetupComponents:
    def test_order_components_called_with_input_configs(self, fakes, make_model):
        a = FakeConfig("a")
        b = FakeConfig("b", conditioned_on_z=["a"])
        fakes.general_utils.order_components.side_effect = lambda cfgs: [a, b]
        model = make_model(b, a)
        fakes.general_utils.order_components.assert_called_once()
        (arg,), _ = fakes.general_utils.order_components.call_args
        assert [c.name for c in arg] == ["b", "a"]
        # Components are built (Component.make) in the order returned by
        # order_components, and the model keeps that order.
        assert [kw["name"] for kw in fakes.component.make_calls] == ["a", "b"]
        assert [c.name for c in model.component_configs] == ["a", "b"]

    def test_make_receives_model_dump_with_conditional_dims(self, fakes, make_model):
        a = FakeConfig("a", latent_dim=2)
        b = FakeConfig("b", latent_dim=3)
        c = FakeConfig(
            "c",
            latent_dim=4,
            conditioned_on_z=["a", "b"],
            conditioned_on_z_hat=["b"],
        )
        model = make_model(a, b, c)
        calls = {kw["name"]: kw for kw in fakes.component.make_calls}
        assert calls["a"]["z_conditional_dims"] is None
        assert calls["a"]["z_hat_conditional_dims"] is None
        # sum of parents' z_prior_parameterizer.event_dims
        assert calls["c"]["z_conditional_dims"] == 2 + 3
        assert calls["c"]["z_hat_conditional_dims"] == 3
        # the rest of model_dump() is forwarded verbatim
        assert calls["c"]["conditioned_on_z"] == ["a", "b"]
        assert calls["c"]["latent_dim"] == 4
        # written to the model's own copy of the config, not the caller's
        (model_c,) = [x for x in model.component_configs if x.name == "c"]
        assert model_c.z_conditional_dims == 5 and model_c.z_hat_conditional_dims == 3
        assert not hasattr(c, "z_conditional_dims")

    def test_setup_components_returns_union_of_modalities_and_vars(self, fakes):
        a = FakeConfig("a", ["rna"], n_vars={"rna": 4}, n_vars_batch_effect={"rna": 2})
        b = FakeConfig(
            "b",
            ["rna", "atac"],
            n_vars={"rna": 4, "atac": 6},
            n_vars_batch_effect={"rna": 2, "atac": 3},
        )
        components, configs, modalities, n_vars, n_vars_be = Model.setup_components(
            [a, b]
        )
        assert set(components) == {"a", "b"}
        assert all(isinstance(c, FakeComponent) for c in components.values())
        assert configs == [a, b]
        assert modalities == {"rna", "atac"}
        assert n_vars == {"rna": 4, "atac": 6}
        assert n_vars_be == {"rna": 2, "atac": 3}


class TestPrepareConditionalDimsConfig:
    def test_no_parents_gives_none(self):
        cfg = FakeConfig("x")
        assert Model.prepare_conditional_dims_config(cfg, {}) == {
            Constants.MODEL_INPUTS_Z_CONDITIONAL_DIMS: None,
            Constants.MODEL_INPUTS_Z_HAT_CONDITIONAL_DIMS: None,
        }

    def test_sums_event_dims_per_kind(self):
        parents = {
            "p": FakeComponent("p", ["rna"], {"rna": 2}, latent_dim=5),
            "q": FakeComponent("q", ["rna"], {"rna": 2}, latent_dim=7),
        }
        cfg = FakeConfig("x", conditioned_on_z=["p"], conditioned_on_z_hat=["p", "q"])
        out = Model.prepare_conditional_dims_config(cfg, parents)
        assert out[Constants.MODEL_INPUTS_Z_CONDITIONAL_DIMS] == 5
        assert out[Constants.MODEL_INPUTS_Z_HAT_CONDITIONAL_DIMS] == 12

    def test_prepare_conditionals_for_dims(self):
        assert list(Model.prepare_conditionals()) == [
            ("conditioned_on_z", "z_conditional_dims"),
            ("conditioned_on_z_hat", "z_hat_conditional_dims"),
        ]

    def test_prepare_conditionals_for_inputs(self):
        zc, zhc = {"a": 1}, {"b": 2}
        assert list(Model.prepare_conditionals(False, zc, zhc)) == [
            ("z_conditional", "conditioned_on_z", zc),
            ("z_hat_conditional", "conditioned_on_z_hat", zhc),
        ]


# ---------------------------------------------------------------------------
# setup_inputs
# ---------------------------------------------------------------------------


class TestSetupInputs:
    def test_keys_and_shapes(self):
        inputs = Model.setup_inputs(
            modality_names=["rna", "atac"],
            n_vars={"rna": 4, "atac": 6},
            n_vars_batch_effect={"rna": 2, "atac": 3},
        )
        assert set(inputs) == {
            "rna_matrix",
            "rna_batch_effect",
            "atac_matrix",
            "atac_batch_effect",
        }
        assert tuple(inputs["rna_matrix"].shape) == (None, 4)
        assert tuple(inputs["atac_matrix"].shape) == (None, 6)
        assert tuple(inputs["rna_batch_effect"].shape) == (None, 2)
        assert tuple(inputs["atac_batch_effect"].shape) == (None, 3)
        assert inputs["rna_matrix"].name == "rna_matrix"


# ---------------------------------------------------------------------------
# prepare_component_inputs
# ---------------------------------------------------------------------------


class TestPrepareComponentInputs:
    def _components(self):
        return {
            "a": FakeComponent("a", ["rna"], {"rna": 2}),
            "c": FakeComponent("c", ["rna", "atac"], {"rna": 2, "atac": 3}),
        }

    def test_selects_matrix_and_batch_effect_of_component_modalities(self):
        batch = make_batch({"rna": 2, "atac": 3, "adt": 1})
        batch["rna_libsize"] = tf.ones((3, 1))
        out = Model.prepare_component_inputs(
            batch, FakeConfig("c", ["rna", "atac"]), "c", self._components()
        )
        assert set(out) == {
            "rna_matrix",
            "rna_batch_effect",
            "atac_matrix",
            "atac_batch_effect",
        }
        for k in out:
            assert out[k] is batch[k]

    def test_modalities_come_from_component_not_config(self):
        # config claims atac, the built component only has rna
        batch = make_batch({"rna": 2, "atac": 3})
        out = Model.prepare_component_inputs(
            batch, FakeConfig("a", ["rna", "atac"]), "a", self._components()
        )
        assert set(out) == {"rna_matrix", "rna_batch_effect"}

    def test_conditionals_select_z_vs_z_hat_and_follow_config_order(self):
        batch = make_batch({"rna": 2})
        z = {"p": tf.fill((3, 1), 1.0), "q": tf.fill((3, 2), 2.0)}
        z_hat = {"p": tf.fill((3, 1), 10.0), "q": tf.fill((3, 2), 20.0)}
        cfg = FakeConfig("a", conditioned_on_z=["q", "p"], conditioned_on_z_hat=["p"])
        out = Model.prepare_component_inputs(
            batch, cfg, "a", self._components(), z, z_hat
        )
        np.testing.assert_array_equal(
            _values(out["z_conditional"]), np.array([[2.0, 2.0, 1.0]] * 3)
        )
        np.testing.assert_array_equal(
            _values(out["z_hat_conditional"]), np.full((3, 1), 10.0)
        )

    def test_no_conditional_keys_without_parents(self):
        batch = make_batch({"rna": 2})
        out = Model.prepare_component_inputs(
            batch, FakeConfig("a"), "a", self._components(), {"p": tf.ones((3, 1))}
        )
        assert "z_conditional" not in out and "z_hat_conditional" not in out

    def test_dict_config_has_no_conditionals(self):
        # encode/decode pass model_dump() dicts; getattr on a dict is None
        batch = make_batch({"rna": 2})
        out = Model.prepare_component_inputs(
            batch,
            {"conditioned_on_z": ["p"]},
            "a",
            self._components(),
            {"p": tf.ones((3, 1))},
        )
        assert set(out) == {"rna_matrix", "rna_batch_effect"}

    def test_default_dicts_not_mutated_between_calls(self):
        defaults = Model.prepare_component_inputs.__defaults__
        batch = make_batch({"rna": 2})
        Model.prepare_component_inputs(batch, FakeConfig("a"), "a", self._components())
        assert defaults[0] == {} and defaults[1] == {}

    def test_missing_batch_keys_become_none(self):
        out = Model.prepare_component_inputs(
            {}, FakeConfig("a"), "a", self._components()
        )
        assert out == {"rna_matrix": None, "rna_batch_effect": None}


# ---------------------------------------------------------------------------
# make / setup_outputs: wiring
# ---------------------------------------------------------------------------


class TestMakeWiring:
    def test_model_attributes(self, make_model):
        a, b = FakeConfig("a"), FakeConfig("b", conditioned_on_z=["a"])
        model = make_model(a, b, name="mymodel")
        assert isinstance(model, Model)
        assert model.name == "mymodel"
        assert set(model.components) == {"a", "b"}
        assert [c.name for c in model.component_configs] == ["a", "b"]
        # the model keeps copies, not the caller's objects
        assert all(c is not x for c, x in zip(model.component_configs, (a, b)))

    def test_component_configs_are_read_only(self, make_model):
        a = FakeConfig("a")
        b = FakeConfig("b", conditioned_on_z=["a"])
        model = make_model(a, b)
        configs = model.component_configs
        assert isinstance(configs, tuple)
        with pytest.raises(TypeError):
            configs[0] = FakeConfig("x")
        # modifying a returned config, or the caller's original, does not
        # change the model
        configs[1].conditioned_on_z.append("x")
        b.conditioned_on_z.append("y")
        assert model.component_configs[1].conditioned_on_z == ["a"]

    def test_default_name(self, make_model):
        assert make_model(FakeConfig("a")).name == "cavachon"

    def test_outputs_are_prefixed_with_component_name(self, make_model):
        model = make_model(FakeConfig("a", ["rna"]))
        out = model(make_batch({"rna": 2}))
        assert set(out) == {
            "a_z",
            "a_z_hat",
            "a_z_parameters",
            "a_z_prior_parameters",
            "a_z_hat_parameters",
            "a_z_hat_prior_parameters",
            "a_rna_x_parameters",
        }

    def test_child_gets_parent_z_and_z_hat(self, make_model):
        a = FakeConfig("a", ["rna"], z_hat_offset=10.0)
        b = FakeConfig(
            "b",
            ["atac"],
            n_vars={"atac": 3},
            conditioned_on_z=["a"],
            conditioned_on_z_hat=["a"],
        )
        model = make_model(a, b)
        out = model(make_batch({"rna": 2, "atac": 3}))
        np.testing.assert_allclose(out["b_z_conditional_echo"], out["a_z"])
        np.testing.assert_allclose(out["b_z_hat_conditional_echo"], out["a_z_hat"])
        # z and z_hat really differ so a swap would be detected
        assert not np.allclose(out["a_z"], out["a_z_hat"])

    def test_z_only_child_has_no_z_hat_conditional(self, make_model):
        a = FakeConfig("a", ["rna"])
        b = FakeConfig("b", ["atac"], n_vars={"atac": 3}, conditioned_on_z_hat=["a"])
        model = make_model(a, b)
        out = model(make_batch({"rna": 2, "atac": 3}))
        assert "b_z_conditional_echo" not in out
        np.testing.assert_allclose(out["b_z_hat_conditional_echo"], out["a_z_hat"])

    def test_multi_parent_concat_order_follows_config(self, make_model):
        a = FakeConfig("a", ["rna"], kernel_value=1.0, latent_dim=1, z_hat_offset=100.0)
        b = FakeConfig("b", ["rna"], kernel_value=2.0, latent_dim=2, z_hat_offset=200.0)
        c = FakeConfig(
            "c",
            ["atac"],
            n_vars={"atac": 3},
            conditioned_on_z=["b", "a"],
            conditioned_on_z_hat=["a", "b"],
        )
        model = make_model(a, b, c)
        out = model(make_batch({"rna": 2, "atac": 3}))
        np.testing.assert_allclose(
            out["c_z_conditional_echo"],
            np.concatenate([out["b_z"], out["a_z"]], axis=-1),
        )
        np.testing.assert_allclose(
            out["c_z_hat_conditional_echo"],
            np.concatenate([out["a_z_hat"], out["b_z_hat"]], axis=-1),
        )

    def test_grandchild_wiring_uses_direct_parent(self, make_model):
        a = FakeConfig("a", ["rna"], z_hat_offset=1.0)
        b = FakeConfig(
            "b", ["rna"], kernel_value=3.0, z_hat_offset=5.0, conditioned_on_z_hat=["a"]
        )
        c = FakeConfig("c", ["rna"], conditioned_on_z_hat=["b"], conditioned_on_z=["a"])
        model = make_model(a, b, c)
        out = model(make_batch({"rna": 2}))
        np.testing.assert_allclose(out["c_z_hat_conditional_echo"], out["b_z_hat"])
        np.testing.assert_allclose(out["c_z_conditional_echo"], out["a_z"])

    def test_component_sees_only_its_modalities_and_conditionals(self, make_model):
        a = FakeConfig("a", ["rna"])
        b = FakeConfig("b", ["atac"], n_vars={"atac": 3}, conditioned_on_z=["a"])
        model = make_model(a, b)
        build_call_b = model.components["b"].recorder.calls[0]
        assert build_call_b["keys"] == [
            "atac_batch_effect",
            "atac_matrix",
            "z_conditional",
        ]
        build_call_a = model.components["a"].recorder.calls[0]
        assert build_call_a["keys"] == ["rna_batch_effect", "rna_matrix"]

    def test_numeric_outputs(self, make_model):
        model = make_model(
            FakeConfig("a", ["rna"], kernel_value=0.5, z_hat_offset=10.0)
        )
        batch = make_batch({"rna": 2})
        out = model(batch)
        x = batch["rna_matrix"].numpy()
        z = x @ np.full((2, 2), 0.5)
        np.testing.assert_allclose(out["a_z"], z)
        np.testing.assert_allclose(out["a_z_hat"], z + 10.0)
        np.testing.assert_allclose(
            out["a_rna_x_parameters"], np.repeat((z + 10).sum(-1, keepdims=True), 2, -1)
        )


# ---------------------------------------------------------------------------
# __setattr__ (trainable propagation)
# ---------------------------------------------------------------------------


class TestTrainablePropagation:
    def test_setting_false_freezes_all_components(self, make_model):
        model = make_model(FakeConfig("a"), FakeConfig("b"))
        model.trainable = False
        assert all(not c.trainable for c in model.components.values())
        assert model.trainable_variables == []

    def test_setting_true_does_not_unfreeze_components(self, make_model):
        model = make_model(FakeConfig("a"), FakeConfig("b"))
        model.components["a"].trainable = False
        model.trainable = True
        assert model.components["a"].trainable is False
        assert model.components["b"].trainable is True

    def test_frozen_component_variables_excluded(self, make_model):
        model = make_model(FakeConfig("a"), FakeConfig("b"))
        model.components["b"].trainable = False
        names = [v.path for v in model.trainable_variables]
        assert len(names) == 1 and "a_rna_encoder" in names[0]
