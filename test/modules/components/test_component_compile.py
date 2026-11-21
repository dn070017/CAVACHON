"""Component.compile(): which losses are created, their names, their
weights, the tf.Variable weight indices and the warnings.

Mocked (see conftest.py): StandardKLDivergence, GMMKLDivergence,
NegativeLogDataLikelihood, GMMDensityLoss and ProgressiveScaler are fakes
patched into cavachon.modules.components.component. The KL / NLL fakes keep
the real constructor contracts: `weight` is a float that is wrapped into a
(fake) ProgressiveScaler with scale=float(weight); the density fake keeps
the weight object it is given, like GMMDensityLoss does.
"""

import warnings

import pytest
import tensorflow as tf

from .conftest import make_component

GMM = "gmm_kl_divergence"
STD = "standard_kl_divergence"
NLL_RNA = "rna_negative_log_data_likelihood"
NLL_ATAC = "atac_negative_log_data_likelihood"


def _compile(c, **kwargs):
    kwargs.setdefault("optimizer", tf.keras.optimizers.SGD(0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.compile(**kwargs)


def test_default_creates_gmm_kl_and_one_nll_per_modality(fakes):
    c = make_component(name="c")
    _compile(c)
    assert list(c.loss) == [GMM, NLL_RNA, NLL_ATAC]
    gmm = c.loss[GMM]
    assert isinstance(gmm, fakes.GMMKL)
    assert gmm.name == GMM
    assert gmm.weight.scale == 1.0
    for m, key in (("rna", NLL_RNA), ("atac", NLL_ATAC)):
        nll = c.loss[key]
        assert isinstance(nll, fakes.NLL)
        assert nll.name == key
        assert nll.dist_x_z == f"dist_{m}"
        assert nll.weight.scale == 1.0
    assert not fakes.StandardKL.instances
    assert not fakes.Density.instances


def test_weights_for_other_component_names_are_ignored(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"other": 2.0}, gmm_kl_weights={"other": 3.0})
    assert STD not in c.loss
    assert c.loss[GMM].weight.scale == 1.0


def test_standard_and_gmm_weights(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 0.5}, gmm_kl_weights={"c": 2.0})
    assert list(c.loss) == [STD, GMM, NLL_RNA, NLL_ATAC]
    assert c.loss[STD].weight.scale == 0.5
    assert c.loss[STD].name == STD
    assert c.loss[GMM].weight.scale == 2.0
    # the indices are the losses' own weights, so changing them changes
    # the losses (as in Model)
    assert c._standard_kl_weights["c"] is c.loss[STD].weight
    assert c._gmm_kl_weights["c"] is c.loss[GMM].weight


def test_standard_only_does_not_add_default_gmm(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0})
    assert STD in c.loss and GMM not in c.loss


def test_data_weights_from_loss_weights_not_forwarded_to_keras(fakes):
    c = make_component(name="c")
    _compile(c, loss_weights={NLL_RNA: 2.5})
    assert c.loss[NLL_RNA].weight_arg == 2.5
    assert c._data_loss_weights["c"]["rna"] is c.loss[NLL_RNA].weight
    assert c.loss[NLL_RNA].weight.scale == 2.5
    assert c.loss[NLL_ATAC].weight.scale == 1.0
    # Keras must not re-apply them on top (they are popped)
    assert c._compile_loss._user_loss_weights is None


def test_recompile_applies_new_weights(fakes):
    # Component rebuilds its losses on every compile (unlike Model, which
    # reuses existing ones); the indices follow the new losses
    c = make_component(name="c")
    _compile(c, gmm_kl_weights={"c": 2.0}, standard_kl_weights={"c": 0.1})
    _compile(c, gmm_kl_weights={"c": 4.0}, standard_kl_weights={"c": 0.3})
    assert c.loss[GMM].weight.scale == 4.0
    assert c.loss[STD].weight.scale == pytest.approx(0.3)
    assert c._gmm_kl_weights["c"] is c.loss[GMM].weight
    assert c._standard_kl_weights["c"] is c.loss[STD].weight


def test_z_hat_density_loss_only_for_child_learning_priors(fakes):
    root = make_component(name="r", learn_z_hat_priors=True)
    _compile(root)
    assert not any("z_hat_gmm_density" in k for k in root.loss)

    child_no_learn = make_component(name="k", z_dims=2)
    _compile(child_no_learn)
    assert not any("z_hat_gmm_density" in k for k in child_no_learn.loss)

    child = make_component(name="child", z_dims=2, learn_z_hat_priors=True)
    _compile(child)
    key = "child_z_hat_gmm_density"
    assert list(child.loss) == [GMM, key, "z_hat_gmm_kl_divergence", NLL_RNA, NLL_ATAC]
    density = child.loss[key]
    assert density.name == key
    assert density.weight is child._z_hat_density_weights["child"]
    assert float(density.weight) == 0.0  # legacy loss, switched off


def test_recompile_resets_density_weight_to_zero_and_reuses_variable(fakes):
    # the z_hat density loss is legacy (replaced by the z_hat GMM KL):
    # always switched off
    c = make_component(name="c", z_dims=2, learn_z_hat_priors=True)
    _compile(c)
    var = c._z_hat_density_weights["c"]
    assert float(var) == 0.0
    var.assign(1.0)
    _compile(c)
    assert c._z_hat_density_weights["c"] is var
    assert float(var) == 0.0
    assert c.loss["c_z_hat_gmm_density"].weight is var


def test_custom_loss_warns_and_is_kept(fakes):
    c = make_component(name="c")
    custom = {"anything": tf.keras.losses.MeanSquaredError()}
    with pytest.warns(RuntimeWarning, match="custom losses"):
        c.compile(optimizer="sgd", loss=custom)
    assert c.loss == custom
    assert not fakes.GMMKL.instances and not fakes.NLL.instances


def test_metrics_warn_and_are_dropped(fakes):
    c = make_component(name="c")
    with pytest.warns(RuntimeWarning, match="metrics"):
        c.compile(optimizer="sgd", metrics=["mae"])
    assert c._compile_metrics is None


def test_compile_does_not_mutate_caller_dicts(fakes):
    c = make_component(name="c")
    sw = {"c": 1.0}
    gw = {"c": 2.0}
    _compile(c, standard_kl_weights=sw, gmm_kl_weights=gw)
    assert gw == {"c": 2.0}
    assert sw == {"c": 1.0}


def test_compile_keeps_caller_loss_weights(fakes):
    # regression: compile() popped entries from the caller's loss_weights
    c = make_component(name="c")
    lw = {NLL_RNA: 2.0}
    _compile(c, loss_weights=lw)
    _compile(c, loss_weights=lw)
    assert lw == {NLL_RNA: 2.0}
    assert c._data_loss_weights["c"]["rna"].scale == 2.0


def test_kl_weight_index_is_tied_to_the_loss_weight(fakes):
    """regression: the indices were standalone Variables never passed to the
    losses. They are now the losses' own weights, as in Model.compile."""
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0}, gmm_kl_weights={"c": 1.0})
    c._gmm_kl_weights["c"].assign(0.25)
    c._standard_kl_weights["c"].assign(0.5)
    one = tf.ones(())
    assert float(c.loss[GMM].weight(one)) == pytest.approx(0.25)
    assert float(c.loss[STD].weight(one)) == pytest.approx(0.5)


def test_data_weight_index_is_tied_to_the_loss_weight(fakes):
    # regression: the data-loss Variable was copied with float() by the loss
    c = make_component(name="c")
    _compile(c)
    c._data_loss_weights["c"]["rna"].assign(0.0)
    y = tf.ones((4, 6))
    # switched off: scale * 1e-16, so the unweighted value can still be
    # recovered for the metric
    assert float(c.loss[NLL_RNA](y, y)) == pytest.approx(7e-16)


def test_zero_weight_creates_the_loss_switched_off(fakes):
    # regression: a weight of 0 created the loss with scale 0 (never
    # switchable, metric 0 / 0). It is now created with scale 1 and pinned
    # to 0, as in Model.compile.
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 0.0}, gmm_kl_weights={"c": 1.0})
    assert c.loss[STD].weight.scale == 1.0
    assert c.loss[STD].weight.pins == [0.0]
    assert c.loss[GMM].weight.pins == []


def test_z_hat_gmm_kl_for_child_learning_priors(fakes):
    # as in Model.compile: scale = z_hat_kl_weight, pinned to 0 at compile
    child = make_component(
        name="c", z_dims=2, learn_z_hat_priors=True, z_hat_kl_weight=2.5
    )
    _compile(child)
    zkl = child.loss["z_hat_gmm_kl_divergence"]
    assert zkl.weight.scale == 2.5 and zkl.weight.pins == [0.0]
    assert child._z_hat_gmm_kl_weights["c"] is zkl.weight

    default = make_component(name="d", z_dims=2, learn_z_hat_priors=True)
    _compile(default)
    assert default.loss["z_hat_gmm_kl_divergence"].weight.scale == 1.0

    off = make_component(
        name="e", z_dims=2, learn_z_hat_priors=True, z_hat_kl_weight=0.0
    )
    _compile(off)
    assert "z_hat_gmm_kl_divergence" not in off.loss  # as in Model

    root = make_component(name="r", learn_z_hat_priors=True)
    _compile(root)
    assert "z_hat_gmm_kl_divergence" not in root.loss
