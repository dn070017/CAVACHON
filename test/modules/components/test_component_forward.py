"""Forward pass of Component: output keys/shapes, parent conditioning on
z / z_hat, z vs z_hat wiring into decoders, training-flag propagation and
the encode() / hierarchical_encode() / decode() API.

Mocked (see conftest.py): all cavachon.layers / cavachon.losses names used
by component.py, plus fake modules.base / preprocessor builders. The
z_hat sampler fake adds Z_HAT_SHIFT (=100) to the loc so that z and z_hat
can never be confused silently. One test runs the real
modules.base.HierarchicalEncoder with its cavachon.layers deps patched.
"""

import numpy as np
import pytest
import tensorflow as tf

import cavachon.modules.components.component as component_module
from cavachon.environment.constants import Constants
from cavachon.modules.base.hierarchical_encoder import HierarchicalEncoder

from .conftest import (
    BATCH,
    MODALITIES,
    N_BATCH_EFFECT,
    N_LATENT,
    N_PRIORS,
    N_VARS,
    Z_HAT_SHIFT,
    make_batch,
    make_component,
)

Z = Constants.MODULE_INPUTS_CONDITIONED_Z
ZH = Constants.MODULE_INPUTS_CONDITIONED_Z_HAT


def _expected_z_params(c, batch):
    x = np.concatenate([batch["rna_matrix"].numpy(), batch["atac_matrix"].numpy()], -1)
    return x @ c.encoder.w.numpy()


# ---------------------------------------------------------------------------
# output structure
# ---------------------------------------------------------------------------
def test_root_output_keys_and_shapes(fakes):
    c = make_component()
    out = c(make_batch(), training=False)
    assert set(out) == {
        "z",
        "z_hat",
        "z_parameters",
        "z_hat_parameters",
        "z_prior_parameters",
        "rna_x_parameters",
        "atac_x_parameters",
    }
    assert out["z"].shape == (BATCH, N_LATENT)
    assert out["z_hat"].shape == (BATCH, N_LATENT)
    assert out["z_parameters"].shape == (BATCH, 2 * N_LATENT)
    assert out["z_hat_parameters"].shape == (BATCH, 2 * N_LATENT)
    assert out["z_prior_parameters"].shape == (1, N_PRIORS, 2 * N_LATENT + 1)
    assert out["rna_x_parameters"].shape == (BATCH, N_VARS["rna"])
    assert out["atac_x_parameters"].shape == (BATCH, N_VARS["atac"])


def test_root_z_hat_is_z_and_hierarchical_encoder_unused(fakes):
    c = make_component()
    batch = make_batch()
    out = c(batch)
    z_params = _expected_z_params(c, batch)
    np.testing.assert_allclose(out["z_parameters"].numpy(), z_params, rtol=1e-5)
    np.testing.assert_allclose(out["z"].numpy(), z_params[:, :N_LATENT], rtol=1e-5)
    np.testing.assert_allclose(out["z_hat"].numpy(), out["z"].numpy())
    np.testing.assert_allclose(
        out["z_hat_parameters"].numpy(), out["z_parameters"].numpy()
    )
    assert c.hierarchical_encoder.training_calls == []
    # the z_hat sampler is never used for a root
    assert c.z_hat_sampler.training_calls == []


def test_child_output_keys_include_z_hat_prior(fakes):
    c = make_component(z_dims=2, z_hat_dims=3, learn_z_hat_priors=True)
    out = c(make_batch(z_dims=2, z_hat_dims=3))
    assert out["z_hat_prior_parameters"].shape == (1, N_PRIORS, 2 * N_LATENT + 1)
    np.testing.assert_allclose(
        out["z_hat_prior_parameters"].numpy(),
        c.z_hat_prior_parameterizer.w.numpy()[None],
    )
    np.testing.assert_allclose(
        out["z_prior_parameters"].numpy(), c.z_prior_parameterizer.w.numpy()[None]
    )


def test_prior_parameterizers_receive_unit_dummy_input(fakes):
    c = make_component(z_dims=2, learn_z_hat_priors=True)
    c(make_batch(z_dims=2))
    for prior in (c.z_prior_parameterizer, c.z_hat_prior_parameterizer):
        assert prior.input_shapes[-1] == (1, 1)


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_batch_dimension_not_hard_coded(fakes, batch_size):
    c = make_component(z_dims=2, z_hat_dims=2)
    out = c(make_batch(z_dims=2, z_hat_dims=2, batch=batch_size))
    for key in ("z", "z_hat", "rna_x_parameters", "atac_x_parameters"):
        assert out[key].shape[0] == batch_size
    assert out["z_prior_parameters"].shape[0] == 1


# ---------------------------------------------------------------------------
# conditioning on parents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_dims, z_hat_dims", [(2, None), (None, 3), (2, 3)])
def test_child_hierarchical_encoder_receives_own_z_and_parent_tensors(
    fakes, z_dims, z_hat_dims
):
    c = make_component(z_dims=z_dims, z_hat_dims=z_hat_dims)
    batch = make_batch(z_dims=z_dims, z_hat_dims=z_hat_dims)
    out = c(batch)
    seen = c.hierarchical_encoder.last_inputs
    expected_keys = {"z"} | ({Z} if z_dims else set()) | ({ZH} if z_hat_dims else set())
    assert set(seen) == expected_keys
    # own z (the sample, not z_hat and not the parameters)
    np.testing.assert_allclose(seen["z"].numpy(), out["z"].numpy())
    if z_dims:
        np.testing.assert_allclose(seen[Z].numpy(), batch[Z].numpy())
    if z_hat_dims:
        np.testing.assert_allclose(seen[ZH].numpy(), batch[ZH].numpy())


def test_child_z_hat_comes_from_hierarchical_encoder_and_z_hat_sampler(fakes):
    c = make_component(z_dims=2, z_hat_dims=3)
    batch = make_batch(z_dims=2, z_hat_dims=3)
    out = c(batch)
    concat = np.concatenate([batch[Z].numpy(), batch[ZH].numpy(), out["z"].numpy()], -1)
    z_hat_params = concat @ c.hierarchical_encoder.w.numpy()
    np.testing.assert_allclose(out["z_hat_parameters"].numpy(), z_hat_params, rtol=1e-5)
    np.testing.assert_allclose(
        out["z_hat"].numpy(), z_hat_params[:, :N_LATENT] + Z_HAT_SHIFT, rtol=1e-5
    )
    # z is untouched by the hierarchical encoder
    np.testing.assert_allclose(
        out["z"].numpy(), _expected_z_params(c, batch)[:, :N_LATENT], rtol=1e-5
    )


def test_conditionals_are_not_fed_to_preprocessor(fakes):
    c = make_component(z_dims=2, z_hat_dims=3)
    c(make_batch(z_dims=2, z_hat_dims=3))
    assert c.preprocessor.input_keys[-1] == ["atac_matrix", "rna_matrix"]


def test_decoders_receive_z_hat_not_z_and_own_batch_effect(fakes):
    c = make_component(z_dims=2)
    batch = make_batch(z_dims=2)
    out = c(batch)
    for m in MODALITIES:
        matrix = c.decoders[m].last_inputs["matrix"].numpy()
        np.testing.assert_allclose(matrix[:, :N_LATENT], out["z_hat"].numpy())
        np.testing.assert_allclose(
            matrix[:, N_LATENT:], batch[f"{m}_batch_effect"].numpy()
        )
        assert matrix.shape[1] == N_LATENT + N_BATCH_EFFECT[m]
    # z_hat really differs from z here (shifted sampler)
    assert not np.allclose(out["z_hat"].numpy(), out["z"].numpy())


def test_decoder_libsize_only_for_modalities_that_have_it(fakes):
    c = make_component()
    batch = make_batch()
    c(batch)
    rna_inputs = c.decoders["rna"].last_inputs
    np.testing.assert_allclose(
        rna_inputs["libsize"].numpy(),
        batch["rna_matrix"].numpy().sum(-1, keepdims=True),
    )
    assert "libsize" not in c.decoders["atac"].last_inputs


def test_real_hierarchical_encoder_concat_order(
    fakes, real_hierarchical_encoder_env, monkeypatch
):
    """With the real HierarchicalEncoder (identity scaler/parameterizer,
    identity r-kernel), z_hat_parameters is literally the concatenation
    [z_conditional, z_hat_conditional, z] built from the component's
    inputs."""
    monkeypatch.setattr(component_module, "HierarchicalEncoder", HierarchicalEncoder)
    c = make_component(z_dims=2, z_hat_dims=2)
    c.hierarchical_encoder.r_network.layers[0].kernel.assign(np.eye(N_LATENT))
    batch = make_batch(z_dims=2, z_hat_dims=2)
    out = c(batch)
    expected = np.concatenate(
        [batch[Z].numpy(), batch[ZH].numpy(), out["z"].numpy()], -1
    )
    np.testing.assert_allclose(out["z_hat_parameters"].numpy(), expected, rtol=1e-6)
    via_api = c.hierarchical_encode(batch, out["z"])
    np.testing.assert_allclose(via_api["z_hat_parameters"].numpy(), expected, rtol=1e-6)


def test_gradient_flows_to_parent_inputs_and_own_encoder(fakes):
    """No stop_gradient inside the component: z_hat depends differentiably
    on both the parent conditionals and the component's own encoder."""
    c = make_component(z_dims=2, z_hat_dims=2)
    batch = make_batch(z_dims=2, z_hat_dims=2)
    zc = tf.Variable(batch[Z])
    zhc = tf.Variable(batch[ZH])
    with tf.GradientTape() as tape:
        out = c({**batch, Z: zc, ZH: zhc}, training=True)
        target = tf.reduce_sum(out["z_hat"])
    g_zc, g_zhc, g_enc = tape.gradient(target, [zc, zhc, c.encoder.w])
    for g in (g_zc, g_zhc, g_enc):
        assert g is not None
        assert float(tf.reduce_sum(tf.abs(g))) > 0


# ---------------------------------------------------------------------------
# training flag propagation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("training", [True, False])
def test_call_propagates_training_flag(fakes, training):
    c = make_component(z_dims=2)
    c(make_batch(z_dims=2), training=training)
    modules = [
        c.preprocessor,
        c.encoder,
        c.z_sampler,
        c.hierarchical_encoder,
        c.z_hat_sampler,
        *c.decoders.values(),
    ]
    for module in modules:
        assert module.training_calls[-1] is training, module.name


@pytest.mark.parametrize("training", [True, False])
def test_split_api_propagates_training_flag(fakes, training):
    c = make_component(z_dims=2)
    batch = make_batch(z_dims=2)
    for module in [
        c.preprocessor,
        c.encoder,
        c.z_sampler,
        c.hierarchical_encoder,
        c.z_hat_sampler,
        *c.decoders.values(),
    ]:
        module.training_calls.clear()
    enc = c.encode(batch, training=training)
    hier = c.hierarchical_encode(batch, enc["z"], training=training)
    c.decode(batch, hier["z_hat"], training=training)
    assert c.encoder.training_calls == [training]
    assert c.z_sampler.training_calls == [training]
    assert c.hierarchical_encoder.training_calls == [training]
    assert c.z_hat_sampler.training_calls == [training]
    assert c.preprocessor.training_calls == [training, training]
    for d in c.decoders.values():
        assert d.training_calls == [training]


# ---------------------------------------------------------------------------
# encode / hierarchical_encode / decode
# ---------------------------------------------------------------------------
def test_split_api_matches_call(fakes):
    c = make_component(z_dims=2, z_hat_dims=3)
    batch = make_batch(z_dims=2, z_hat_dims=3)
    full = c(batch)
    enc = c.encode(batch)
    assert set(enc) == {"z", "z_parameters"}
    hier = c.hierarchical_encode(batch, enc["z"])
    assert set(hier) == {"z_hat", "z_hat_parameters"}
    dec = c.decode(batch, hier["z_hat"])
    assert set(dec) == {"rna_x_parameters", "atac_x_parameters"}
    for key, value in {**enc, **hier, **dec}.items():
        np.testing.assert_allclose(
            value.numpy(), full[key].numpy(), rtol=1e-5, err_msg=key
        )


def test_hierarchical_encode_root_returns_z_only(fakes):
    c = make_component()
    z = tf.constant(np.ones((BATCH, N_LATENT), np.float32))
    out = c.hierarchical_encode(make_batch(), z)
    assert set(out) == {"z_hat"}
    assert out["z_hat"] is z


def test_encode_missing_matrix_raises(fakes):
    c = make_component()
    batch = make_batch()
    batch.pop("atac_matrix")
    with pytest.raises(ValueError, match="atac_matrix"):
        c.encode(batch)


def test_decode_missing_batch_effect_raises(fakes):
    c = make_component()
    batch = make_batch()
    batch.pop("rna_batch_effect")
    with pytest.raises(ValueError, match="rna_batch_effect"):
        c.decode(batch, tf.zeros((BATCH, N_LATENT)))


def test_decode_missing_decoder_raises(fakes):
    c = make_component()
    c.decoders = {"rna": c.decoders["rna"]}
    with pytest.raises(ValueError, match="atac"):
        c.decode(make_batch(), tf.zeros((BATCH, N_LATENT)))


@pytest.mark.parametrize(
    "method, name", [("hierarchical_encode", "z"), ("decode", "z_hat")]
)
def test_rank_assertions(fakes, method, name):
    # NOTE: the docstrings promise tf.errors.InvalidArgumentError, but for a
    # statically known rank tf.debugging.assert_rank raises ValueError.
    c = make_component(z_dims=2)
    with pytest.raises(ValueError, match=f"Expected '{name}' to have rank 2"):
        getattr(c, method)(make_batch(z_dims=2), tf.zeros((BATCH, N_LATENT, 1)))


def test_decode_uses_given_z_hat(fakes):
    c = make_component()
    batch = make_batch()
    z_hat = tf.constant(np.full((BATCH, N_LATENT), 2.0, np.float32))
    out = c.decode(batch, z_hat)
    be = batch["atac_batch_effect"].numpy()
    expected = np.concatenate([z_hat.numpy(), be], -1) @ c.decoders["atac"].w.numpy()
    np.testing.assert_allclose(out["atac_x_parameters"].numpy(), expected, rtol=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: decode() needs the modality matrices too (preprocessor rerun) "
    "but validates/documents only batch-effect keys",
)
def test_decode_with_documented_keys_only(fakes):
    c = make_component()
    batch = make_batch()
    only_batch_effect = {k: v for k, v in batch.items() if k.endswith("batch_effect")}
    try:
        c.decode(only_batch_effect, tf.zeros((BATCH, N_LATENT)))
    except ValueError as e:  # a clear validation error naming the key is fine
        assert "matrix" in str(e)
