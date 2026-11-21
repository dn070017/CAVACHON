"""Builder functions, make(), constructor attributes, static helpers and
the progressive-scaler hooks of Component.

Mocked (see conftest.py): every cavachon.layers / cavachon.losses name that
component.py imports, plus the modules.base / preprocessor builders, all
patched in the cavachon.modules.components.component namespace. The real
HierarchicalEncoder is used only with its cavachon.layers dependencies
patched in cavachon.modules.base.hierarchical_encoder.
"""

import numpy as np
import pytest
import tensorflow as tf

import cavachon.modules.components.component as component_module
from cavachon.environment.constants import Constants
from cavachon.modules.base.hierarchical_encoder import HierarchicalEncoder
from cavachon.modules.components.component import Component

from .conftest import (
    BATCH,
    DISTRIBUTIONS,
    MODALITIES,
    N_BATCH_EFFECT,
    N_LATENT,
    N_PRIORS,
    N_VARS,
    make_batch,
    make_component,
)

Z = Constants.MODULE_INPUTS_CONDITIONED_Z
ZH = Constants.MODULE_INPUTS_CONDITIONED_Z_HAT


# ---------------------------------------------------------------------------
# setup_inputs
# ---------------------------------------------------------------------------
def test_setup_inputs_root_keys_shapes_and_names():
    inputs = Component.setup_inputs(MODALITIES, N_VARS, N_BATCH_EFFECT)
    assert list(inputs) == [
        "rna_matrix",
        "rna_batch_effect",
        "atac_matrix",
        "atac_batch_effect",
    ]
    for m in MODALITIES:
        x = inputs[f"{m}_matrix"]
        b = inputs[f"{m}_batch_effect"]
        assert tuple(x.shape) == (None, N_VARS[m])
        assert tuple(b.shape) == (None, N_BATCH_EFFECT[m])
        assert x.name == f"{m}_matrix"
        assert b.name == f"{m}_batch_effect"


def test_setup_inputs_conditionals_have_unbounded_batch_dimension():
    inputs = Component.setup_inputs(
        ["rna"], N_VARS, N_BATCH_EFFECT, z_conditional_dims=3, z_hat_conditional_dims=4
    )
    assert tuple(inputs[Z].shape) == (None, 3)
    assert tuple(inputs[ZH].shape) == (None, 4)
    assert inputs[Z].name == Z
    assert inputs[ZH].name == ZH


def test_setup_inputs_only_z_hat_conditional():
    inputs = Component.setup_inputs(
        ["rna"], N_VARS, N_BATCH_EFFECT, z_hat_conditional_dims=2
    )
    assert Z not in inputs
    assert ZH in inputs


# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------
def test_setup_preprocessor_forwards_arguments(fakes):
    p = Component.setup_preprocessor(
        modality_names=MODALITIES,
        distribution_names=DISTRIBUTIONS,
        n_vars=N_VARS,
        n_dims=13,
        name="pp",
        unrelated="ignored",
    )
    assert isinstance(p, fakes.Preprocessor)
    assert p.modality_names == MODALITIES
    assert p.distribution_names == DISTRIBUTIONS
    assert p.n_vars == N_VARS
    assert p.n_dims == 13
    assert p.name == "pp"


def test_setup_encoder_forwards_arguments(fakes):
    e = Component.setup_encoder(n_latent_dims=4, n_encoder_layers=2, name="enc")
    assert isinstance(e, fakes.Encoder)
    assert (e.n_latent_dims, e.n_layers, e.name) == (4, 2, "enc")


def test_setup_decoders_int_layers(fakes):
    decoders = Component.setup_decoders(
        MODALITIES, DISTRIBUTIONS, N_VARS, n_decoder_layers=2, name="c"
    )
    assert list(decoders) == MODALITIES
    for m in MODALITIES:
        d = decoders[m]
        assert d.distribution_name == DISTRIBUTIONS[m]
        assert d.n_vars == N_VARS[m]
        assert d.n_layers == 2
        assert d.name == f"c_{m}"


def test_setup_decoders_mapping_layers_fall_back_to_three(fakes):
    decoders = Component.setup_decoders(
        MODALITIES, DISTRIBUTIONS, N_VARS, n_decoder_layers={"rna": 5}, name="c"
    )
    assert decoders["rna"].n_layers == 5
    assert decoders["atac"].n_layers == 3


def test_setup_decoders_does_not_mutate_mapping(fakes):
    layers = {"rna": 5}
    Component.setup_decoders(MODALITIES, DISTRIBUTIONS, N_VARS, n_decoder_layers=layers)
    assert layers == {"rna": 5}


def test_setup_prior_parameterizers(fakes):
    zp = Component.setup_z_prior_parameterizer(
        n_latent_dims=3, n_latent_priors=7, name="zp"
    )
    zhp = Component.setup_z_hat_prior_parameterizer(
        n_latent_dims=4, n_latent_priors=9, name="zhp"
    )
    assert (zp.event_dims, zp.n_components, zp.name) == (3, 7, "zp")
    assert (zhp.event_dims, zhp.n_components, zhp.name) == (4, 9, "zhp")


def test_samplers_z_stochastic_and_z_hat_deterministic(fakes):
    zs = Component.setup_z_sampler(name="zs")
    zhs = Component.setup_z_hat_sampler(name="zhs")
    assert zs.stochastic is True and zs.name == "zs"
    assert zhs.stochastic is False and zhs.name == "zhs"


def test_setup_hierarchical_encoder_forwards_flags_and_use_bias(fakes):
    h = Component.setup_hierarchical_encoder(
        n_latent_dims=3,
        is_conditioned_on_z=True,
        is_conditioned_on_z_hat=False,
        use_bias=True,
        name="he",
    )
    assert isinstance(h, fakes.HierarchicalEncoder)
    assert h.n_latent_dims == 3
    assert h.is_conditioned_on_z is True
    assert h.is_conditioned_on_z_hat is False
    assert h.use_bias is True
    assert h.name == "he"


def test_setup_hierarchical_encoder_use_bias_defaults_false(fakes):
    assert Component.setup_hierarchical_encoder().use_bias is False


def test_setup_hierarchical_encoder_forwards_progressive_iterations(fakes):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    h = Component.setup_hierarchical_encoder(progressive_iterations=123)
    assert h.progressive_iterations == 123


def test_real_hierarchical_encoder_gets_progressive_iterations(
    fakes, real_hierarchical_encoder_env, monkeypatch
):
    # regression: fixed bug, see test_report/UNIT_TEST_REPORT.md
    monkeypatch.setattr(component_module, "HierarchicalEncoder", HierarchicalEncoder)
    h = Component.setup_hierarchical_encoder(progressive_iterations=123)
    assert float(h.progressive_scaler.total_iterations) == 123.0


# ---------------------------------------------------------------------------
# make()
# ---------------------------------------------------------------------------
def test_make_root_wires_builders_and_names(fakes):
    c = make_component(name="root")
    assert isinstance(c, Component)
    assert c.name == "root"
    assert list(c.modality_names) == MODALITIES
    assert dict(c.distribution_names) == DISTRIBUTIONS
    assert c.encoder.name == "root_encoder"
    assert c.preprocessor.name == "root_preprocessor"
    assert c.hierarchical_encoder.name == "root_hierarchical_encoder"
    assert c.z_prior_parameterizer.name == "root_z_prior_parameterizer"
    assert c.z_sampler.name == "root_z_sampler"
    assert c.z_hat_sampler.name == "root_z_hat_sampler"
    assert {m: d.name for m, d in c.decoders.items()} == {
        "rna": "root_rna",
        "atac": "root_atac",
    }
    # the preprocessor reduces to the widest encoder layer
    assert c.preprocessor.n_dims == c.encoder.max_n_neurons == 7
    assert (
        c.z_prior_parameterizer.event_dims,
        c.z_prior_parameterizer.n_components,
    ) == (
        N_LATENT,
        N_PRIORS,
    )
    assert c.z_hat_prior_parameterizer is None
    assert c.learn_z_hat_priors is False
    assert c.is_root
    assert list(c.conditioned_on_z) == [] and list(c.conditioned_on_z_hat) == []


def test_make_root_forces_learn_z_hat_priors_off(fakes):
    c = make_component(learn_z_hat_priors=True)
    assert c.learn_z_hat_priors is False
    assert c.z_hat_prior_parameterizer is None
    assert len(fakes.Prior.instances) == 1


def test_make_child_with_learned_z_hat_prior(fakes):
    c = make_component(name="child", z_hat_dims=2, learn_z_hat_priors=True)
    assert c.learn_z_hat_priors is True
    zhp = c.z_hat_prior_parameterizer
    assert zhp is not None and zhp is not c.z_prior_parameterizer
    assert zhp.name == "child_z_hat_prior_parameterizer"
    assert (zhp.event_dims, zhp.n_components) == (N_LATENT, N_PRIORS)
    assert not c.is_root
    assert list(c.conditioned_on_z_hat) == ["parent_z_hat"]


def test_make_learn_z_hat_priors_none_is_false(fakes):
    c = make_component(z_dims=2, learn_z_hat_priors=None)
    assert c.learn_z_hat_priors is False
    assert c.z_hat_prior_parameterizer is None


@pytest.mark.parametrize(
    "z_dims, z_hat_dims, flags",
    [
        (2, None, (True, False)),
        (None, 3, (False, True)),
        (2, 3, (True, True)),
        (None, None, (False, False)),
    ],
)
def test_make_hierarchical_encoder_condition_flags(fakes, z_dims, z_hat_dims, flags):
    c = make_component(z_dims=z_dims, z_hat_dims=z_hat_dims)
    h = c.hierarchical_encoder
    assert (h.is_conditioned_on_z, h.is_conditioned_on_z_hat) == flags


@pytest.mark.parametrize("reparam, use_bias", [(True, False), (False, True)])
def test_make_use_bias_is_not_reparameterize_z_hat(fakes, reparam, use_bias):
    c = make_component(z_dims=2, reparameterize_z_hat=reparam)
    assert c.hierarchical_encoder.use_bias is use_bias
    assert c.reparameterize_z_hat is reparam


def test_make_default_reparameterize_gives_no_bias(fakes):
    c = make_component(z_dims=2)
    assert c.hierarchical_encoder.use_bias is False
    assert c.reparameterize_z_hat is True


def test_make_with_real_hierarchical_encoder_use_bias(
    fakes, real_hierarchical_encoder_env, monkeypatch
):
    monkeypatch.setattr(component_module, "HierarchicalEncoder", HierarchicalEncoder)
    c = make_component(z_dims=2, reparameterize_z_hat=False)
    r_dense = c.hierarchical_encoder.r_network.layers[0]
    assert r_dense.use_bias is True


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------
def test_conditioned_lists_are_not_shared_between_instances(fakes):
    a = make_component(name="a")
    b = make_component(name="b")
    assert a.conditioned_on_z is not b.conditioned_on_z
    a.conditioned_on_z.append("x")
    assert list(b.conditioned_on_z) == []


@pytest.mark.parametrize(
    "z, zh, root", [([], [], True), (["p"], [], False), ([], ["p"], False)]
)
def test_is_root_property(fakes, z, zh, root):
    c = make_component()
    c.conditioned_on_z = z
    c.conditioned_on_z_hat = zh
    assert c.is_root is root


def test_is_root_inputs():
    assert Component.is_root_inputs({"rna_matrix": 1})
    assert not Component.is_root_inputs({Z: 1})
    assert not Component.is_root_inputs({ZH: 1})


# ---------------------------------------------------------------------------
# progressive scaler hooks
# ---------------------------------------------------------------------------
def test_set_progressive_scaler_iteration_assigns_floats(fakes):
    c = make_component(z_dims=2)
    c.set_progressive_scaler_iteration(3, 10)
    ps = c.hierarchical_encoder.progressive_scaler
    assert float(ps.current_iteration) == 3.0
    assert float(ps.total_iterations) == 10.0
    c.set_progressive_scaler_iteration()
    assert (float(ps.current_iteration), float(ps.total_iterations)) == (1.0, 1.0)


def test_trainable_false_pins_alpha_to_one(fakes):
    c = make_component(z_dims=2)
    c.set_progressive_scaler_iteration(3, 10)
    c.trainable = False
    ps = c.hierarchical_encoder.progressive_scaler
    assert (float(ps.current_iteration), float(ps.total_iterations)) == (1.0, 1.0)
    assert c.trainable is False


def test_trainable_true_keeps_alpha(fakes):
    c = make_component(z_dims=2)
    c.set_progressive_scaler_iteration(3, 10)
    c.trainable = True
    ps = c.hierarchical_encoder.progressive_scaler
    assert (float(ps.current_iteration), float(ps.total_iterations)) == (3.0, 10.0)


def test_other_attributes_do_not_touch_alpha(fakes):
    c = make_component(z_dims=2)
    c.set_progressive_scaler_iteration(3, 10)
    c.some_flag = False
    ps = c.hierarchical_encoder.progressive_scaler
    assert float(ps.current_iteration) == 3.0


# ---------------------------------------------------------------------------
# static helpers
# ---------------------------------------------------------------------------
def test_prepare_preprocessor_inputs_drops_conditionals_and_batch_effects():
    batch = make_batch(z_dims=2, z_hat_dims=2)
    original_keys = set(batch)
    out = Component.prepare_preprocessor_inputs(batch, MODALITIES)
    assert set(out) == {"rna_matrix", "atac_matrix"}
    assert out["rna_matrix"] is batch["rna_matrix"]
    assert set(batch) == original_keys  # input not mutated


def test_prepare_preprocessor_inputs_keeps_other_modalities_batch_effect():
    batch = make_batch()
    out = Component.prepare_preprocessor_inputs(batch, ["rna"])
    assert "atac_batch_effect" in out and "rna_batch_effect" not in out


def test_prepare_hierarchical_encoder_inputs():
    batch = make_batch(z_dims=2, z_hat_dims=3)
    z = tf.ones((BATCH, N_LATENT))
    out = Component.prepare_hierarchical_encoder_inputs(batch, z)
    assert set(out) == {Constants.MODEL_OUTPUTS_Z, Z, ZH}
    assert out[Constants.MODEL_OUTPUTS_Z] is z
    assert out[Z] is batch[Z]
    assert out[ZH] is batch[ZH]
    root = Component.prepare_hierarchical_encoder_inputs(make_batch(), z)
    assert set(root) == {Constants.MODEL_OUTPUTS_Z}


def test_prepare_decoder_inputs_concat_order_and_libsize():
    batch = make_batch()
    z_hat = tf.constant(
        np.arange(BATCH * N_LATENT, dtype=np.float32).reshape(BATCH, N_LATENT)
    )
    libsize = tf.fill((BATCH, 1), 3.0)
    out = Component.prepare_decoder_inputs(
        batch, "rna", z_hat, {"rna_matrix_libsize": libsize}
    )
    expected = np.concatenate([z_hat.numpy(), batch["rna_batch_effect"].numpy()], -1)
    np.testing.assert_allclose(out["matrix"].numpy(), expected)
    assert out["libsize"] is libsize


def test_prepare_decoder_inputs_without_libsize_and_default_not_shared():
    batch = make_batch()
    z_hat = tf.zeros((BATCH, N_LATENT))
    out1 = Component.prepare_decoder_inputs(batch, "atac", z_hat)
    out2 = Component.prepare_decoder_inputs(batch, "atac", z_hat)
    assert set(out1) == set(out2) == {"matrix"}
    assert out1["matrix"].shape == (BATCH, N_LATENT + N_BATCH_EFFECT["atac"])
    # other modality's libsize is not picked up
    out3 = Component.prepare_decoder_inputs(
        batch, "atac", z_hat, {"rna_matrix_libsize": tf.ones((BATCH, 1))}
    )
    assert "libsize" not in out3
