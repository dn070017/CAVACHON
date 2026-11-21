"""Component.train_step / test_step: loss aggregation and weighting,
what tensors feed which loss (z vs z_hat), reported (unweighted) metrics,
which variables are trained, and phase-wise freezing.

Mocked (see conftest.py): all losses are fakes with per-sample value
`weight * (value + coef * mean(y_pred))` (GMM KL 5, standard KL 3, NLL 7,
z_hat density 11 by default); ProgressiveScaler, ToDense, samplers and
priors are fakes patched into cavachon.modules.components.component. The
optimizer is real Keras SGD.
"""

import warnings

import numpy as np
import pytest
import tensorflow as tf


from .conftest import N_LATENT, Z_HAT_SHIFT, make_batch, make_component

GMM = "gmm_kl_divergence"
STD = "standard_kl_divergence"
NLL_RNA = "rna_negative_log_data_likelihood"
NLL_ATAC = "atac_negative_log_data_likelihood"


def _compile(c, lr=0.1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.compile(optimizer=tf.keras.optimizers.SGD(lr), **kwargs)


def _train(c, batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {k: float(v) for k, v in c.train_step(batch).items()}


# ---------------------------------------------------------------------------
# Keras 3 compatibility
# ---------------------------------------------------------------------------
def test_train_step_runs_on_keras3_without_patching(fakes):
    # regression: train_step called self.compiled_metrics.update_state(y_true,
    # y_pred), which raises on Keras 3 (Model.train_step never used it)
    c = make_component(name="c")
    _compile(c)
    logs = _train(c, make_batch())
    assert logs["loss"] == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# aggregation and weighting
# ---------------------------------------------------------------------------
def test_default_loss_is_sum_of_gmm_and_nll(fakes):
    c = make_component(name="c")
    _compile(c)
    logs = _train(c, make_batch())
    assert logs == pytest.approx(
        {
            "loss": 5.0 + 7.0 + 7.0,
            GMM: 5.0,
            NLL_RNA: 7.0,
            NLL_ATAC: 7.0,
        }
    )


def test_weights_applied_once_and_metrics_unweighted(fakes):
    c = make_component(name="c")
    _compile(
        c,
        standard_kl_weights={"c": 0.5},
        gmm_kl_weights={"c": 2.0},
        loss_weights={NLL_RNA: 2.5},
    )
    logs = _train(c, make_batch())
    assert logs["loss"] == pytest.approx(0.5 * 3 + 2.0 * 5 + 2.5 * 7 + 7)
    assert logs[STD] == pytest.approx(3.0)
    assert logs[GMM] == pytest.approx(5.0)
    assert logs[NLL_RNA] == pytest.approx(7.0)
    assert logs[NLL_ATAC] == pytest.approx(7.0)
    # each loss is evaluated once, for both the total and its metric
    for loss_fn in c.loss.values():
        assert len(loss_fn.calls) == 1


def test_z_hat_density_weight_variable_and_metric(fakes):
    c = make_component(name="c", z_dims=2, learn_z_hat_priors=True)
    _compile(c)
    var = c._z_hat_density_weights["c"]
    var.assign(0.5)
    logs = _train(c, make_batch(z_dims=2))
    assert logs["loss"] == pytest.approx(5 + 0.5 * 11 + 7 + 7)
    assert logs["c_z_hat_gmm_density"] == pytest.approx(11.0)
    # the temporary weight swap used for the metric is undone
    assert c.loss["c_z_hat_gmm_density"].weight is var


def test_results_are_stable_across_calls(fakes):
    c = make_component(name="c", z_dims=2, learn_z_hat_priors=True)
    _compile(c)
    batch = make_batch(z_dims=2)
    first = _train(c, batch)
    second = _train(c, batch)
    assert first == pytest.approx(second)
    assert set(first) == {
        "loss",
        GMM,
        "c_z_hat_gmm_density",
        "z_hat_gmm_kl_divergence",
        NLL_RNA,
        NLL_ATAC,
    }


# ---------------------------------------------------------------------------
# wiring of y_true / y_pred
# ---------------------------------------------------------------------------
def test_loss_inputs_root(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0}, gmm_kl_weights={"c": 1.0})
    batch = make_batch()
    out = c(batch, training=True)
    _train(c, batch)
    kl_pred = np.concatenate([out["z"].numpy(), out["z_parameters"].numpy()], -1)
    for key in (STD, GMM):
        y_true, y_pred = c.loss[key].calls[0]
        np.testing.assert_allclose(y_true.numpy(), out["z_prior_parameters"].numpy())
        np.testing.assert_allclose(y_pred.numpy(), kl_pred, rtol=1e-5)
        assert y_pred.shape[-1] == 3 * N_LATENT
    for m, key in (("rna", NLL_RNA), ("atac", NLL_ATAC)):
        y_true, y_pred = c.loss[key].calls[0]
        np.testing.assert_allclose(y_true.numpy(), batch[f"{m}_matrix"].numpy())
        np.testing.assert_allclose(
            y_pred.numpy(), out[f"{m}_x_parameters"].numpy(), rtol=1e-5
        )
    assert fakes.ToDense.calls == ["rna_matrix", "atac_matrix"]


def test_loss_inputs_child_kl_on_z_density_on_z_hat(fakes):
    c = make_component(name="c", z_dims=2, z_hat_dims=2, learn_z_hat_priors=True)
    _compile(c)
    batch = make_batch(z_dims=2, z_hat_dims=2)
    zhp = c.z_hat_prior_parameterizer
    zhp.w.assign(zhp.w + 3.0)  # make the two priors distinguishable
    out = c(batch, training=True)
    _train(c, batch)
    kl_true, kl_pred = c.loss[GMM].calls[0]
    np.testing.assert_allclose(kl_true.numpy(), out["z_prior_parameters"].numpy())
    _, kl_pred = c.loss[GMM].calls[0]
    # KL is on z (no z_hat shift), never on z_hat
    np.testing.assert_allclose(
        kl_pred.numpy()[:, :N_LATENT], out["z"].numpy(), rtol=1e-5
    )
    assert np.all(np.abs(kl_pred.numpy()) < Z_HAT_SHIFT / 2)
    d_true, d_pred = c.loss["c_z_hat_gmm_density"].calls[0]
    np.testing.assert_allclose(d_pred.numpy(), out["z_hat"].numpy(), rtol=1e-5)
    np.testing.assert_allclose(d_true.numpy(), out["z_hat_prior_parameters"].numpy())
    assert not np.allclose(d_true.numpy(), out["z_prior_parameters"].numpy())


def test_standard_kl_only_trains(fakes):
    # regression: with only standard_kl_weights, train_step fed y only
    # under the GMM key, so compute_loss raised KeyError for the standard KL
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0})
    logs = _train(c, make_batch())
    assert logs[STD] == pytest.approx(3.0)
    assert logs["loss"] == pytest.approx(3 + 7 + 7)


# ---------------------------------------------------------------------------
# gradients and trainable variables
# ---------------------------------------------------------------------------
def _expected_loss(c, batch, density=False):
    out = c(batch, training=True)
    kl_pred = tf.concat([out["z"], out["z_parameters"]], -1)
    total = tf.reduce_mean(5.0 + tf.reduce_mean(kl_pred, -1))
    for m in ("rna", "atac"):
        total += tf.reduce_mean(7.0 + tf.reduce_mean(out[f"{m}_x_parameters"], -1))
    if density:
        total += tf.reduce_mean(11.0 + tf.reduce_mean(out["z_hat"], -1))
    return total


@pytest.mark.parametrize("child", [False, True])
def test_sgd_update_matches_gradient_of_total_loss(fakes, child):
    for cls in (fakes.GMMKL, fakes.NLL, fakes.Density):
        cls.coef = 1.0
    kw = dict(z_dims=2, learn_z_hat_priors=True) if child else {}
    c = make_component(name="c", **kw)
    batch = make_batch(z_dims=2 if child else None)
    _compile(c, lr=0.1)
    variables = list(c.trainable_variables)
    with tf.GradientTape() as tape:
        # the legacy z_hat density loss is compiled but switched off
        expected = _expected_loss(c, batch, density=False)
    grads = tape.gradient(expected, variables)
    before = [v.numpy().copy() for v in variables]
    logs = _train(c, batch)
    assert logs["loss"] == pytest.approx(float(expected), rel=1e-5)
    for v, b, g in zip(variables, before, grads):
        g = np.zeros_like(b) if g is None else g.numpy()
        np.testing.assert_allclose(
            v.numpy(), b - 0.1 * g, rtol=1e-4, atol=1e-6, err_msg=v.path
        )


def test_root_trainable_variables_exclude_unused_modules(fakes):
    c = make_component(name="c")
    paths = sorted(v.path for v in c.trainable_variables)
    assert paths == sorted(
        ["c_encoder/w", "c_rna/w", "c_atac/w", "c_z_prior_parameterizer/w"]
    )


def test_child_trains_hierarchical_encoder_and_z_hat_prior(fakes):
    for cls in (fakes.GMMKL, fakes.NLL, fakes.Density):
        cls.coef = 1.0
    c = make_component(name="c", z_dims=2, learn_z_hat_priors=True)
    _compile(c)
    paths = {v.path for v in c.trainable_variables}
    assert {"c_hierarchical_encoder/w", "c_z_hat_prior_parameterizer/w"} <= paths
    h_before = c.hierarchical_encoder.w.numpy().copy()
    _train(c, make_batch(z_dims=2))
    assert not np.allclose(c.hierarchical_encoder.w.numpy(), h_before)


def test_frozen_encoder_is_not_updated(fakes):
    for cls in (fakes.GMMKL, fakes.NLL):
        cls.coef = 1.0
    c = make_component(name="c")
    c.encoder.trainable = False
    _compile(c)
    assert "c_encoder/w" not in {v.path for v in c.trainable_variables}
    enc_before = c.encoder.w.numpy().copy()
    dec_before = c.decoders["rna"].w.numpy().copy()
    _train(c, make_batch())
    np.testing.assert_array_equal(c.encoder.w.numpy(), enc_before)
    assert not np.allclose(c.decoders["rna"].w.numpy(), dec_before)


def test_frozen_component_has_no_trainable_variables(fakes):
    c = make_component(name="c", z_dims=2)
    c.trainable = False
    assert c.trainable_variables == []
    c.trainable = True
    assert len(c.trainable_variables) == 5


# ---------------------------------------------------------------------------
# test_step
# ---------------------------------------------------------------------------
def test_test_step_reports_same_losses(fakes):
    # regression: train_step was overridden but test_step was not, so
    # Keras' default test_step could not match the loss dict
    c = make_component(name="c")
    _compile(c)
    batch = make_batch()
    before = [v.numpy().copy() for v in c.trainable_variables]
    logs = c.test_step(batch)
    assert {k: float(v) for k, v in logs.items()} == pytest.approx(
        {"loss": 19.0, GMM: 5.0, NLL_RNA: 7.0, NLL_ATAC: 7.0}
    )
    # no training
    for v, b in zip(c.trainable_variables, before):
        np.testing.assert_array_equal(v.numpy(), b)


def test_zero_weight_loss_is_reported_but_not_trained(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 0.0}, gmm_kl_weights={"c": 1.0})
    logs = _train(c, make_batch())
    assert logs[STD] == pytest.approx(3.0)  # unweighted value reported
    assert logs["loss"] == pytest.approx(5 + 7 + 7)  # not in the total


# ---------------------------------------------------------------------------
# NaN gradients
# ---------------------------------------------------------------------------
def test_gradients_pass_through_remove_nan_gradients(fakes):
    # regression: Component.train_step applied raw gradients, while
    # Model.train_step drops NaN gradients via TensorUtils
    c = make_component(name="c")
    _compile(c)
    _train(c, make_batch())
    fakes.TensorUtils.remove_nan_gradients.assert_called_once()
    (grads,), _ = fakes.TensorUtils.remove_nan_gradients.call_args
    assert len(list(grads)) == len(c.trainable_variables)


def test_cleaned_gradients_are_the_ones_applied(fakes):
    c = make_component(name="c")
    _compile(c, lr=1.0)
    fakes.TensorUtils.remove_nan_gradients.side_effect = lambda gradients: [
        None if g is None else tf.zeros_like(g) for g in gradients
    ]
    before = [v.numpy().copy() for v in c.trainable_variables]
    _train(c, make_batch())
    for b, v in zip(before, c.trainable_variables):
        np.testing.assert_array_equal(b, v.numpy())


def test_z_hat_gmm_kl_inputs_and_switching_on(fakes):
    c = make_component(name="c", z_dims=2, learn_z_hat_priors=True)
    _compile(c)
    batch = make_batch(z_dims=2)
    zkl = "z_hat_gmm_kl_divergence"
    off = _train(c, batch)
    # pinned to 0: reported, not in the total
    assert off[zkl] == pytest.approx(5.0)
    assert off["loss"] == pytest.approx(5 + 7 + 7)
    y_true, y_pred = c.loss[zkl].calls[0]
    out = c(batch, training=True)
    np.testing.assert_allclose(y_true.numpy(), out["z_hat_prior_parameters"].numpy())
    assert (
        y_pred.shape[-1] == out["z_hat"].shape[-1] + out["z_hat_parameters"].shape[-1]
    )
    # switched on in standalone training with pin_to
    c._z_hat_gmm_kl_weights["c"].pin_to(1.0)
    on = _train(c, batch)
    assert on["loss"] == pytest.approx(5 + 5 + 7 + 7)


def test_extra_batch_entries_are_ignored(fakes):
    # regression: like Model, the batch is restricted to the component's
    # inputs, so a dataset with other modalities can be used directly
    c = make_component(name="c", modalities=["rna"])
    _compile(c)
    batch = make_batch(modalities=["rna"])
    full = dict(make_batch(), other_matrix=tf.zeros((4, 9)))
    assert _train(c, full) == pytest.approx(_train(c, batch))
    assert {k: float(v) for k, v in c.test_step(full).items()} == pytest.approx(
        {k: float(v) for k, v in c.test_step(batch).items()}
    )
    np.testing.assert_allclose(
        c.predict_step(full)["z"].numpy(), c.predict_step(batch)["z"].numpy()
    )
