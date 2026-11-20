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
    assert float(c._standard_kl_weights["c"]) == 0.5
    assert float(c._gmm_kl_weights["c"]) == 2.0
    assert c._standard_kl_weights["c"].trainable is False


def test_standard_only_does_not_add_default_gmm(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0})
    assert STD in c.loss and GMM not in c.loss


def test_data_weights_from_loss_weights_not_forwarded_to_keras(fakes):
    c = make_component(name="c")
    _compile(c, loss_weights={NLL_RNA: 2.5})
    assert c.loss[NLL_RNA].weight_arg is c._data_loss_weights["c"]["rna"]
    assert float(c._data_loss_weights["c"]["rna"]) == 2.5
    assert float(c._data_loss_weights["c"]["atac"]) == 1.0
    assert c.loss[NLL_RNA].weight.scale == 2.5
    # Keras must not re-apply them on top (they are popped)
    assert c._compile_loss._user_loss_weights is None


def test_recompile_reuses_kl_weight_variables(fakes):
    c = make_component(name="c")
    _compile(c, gmm_kl_weights={"c": 2.0}, standard_kl_weights={"c": 0.1})
    gmm_var = c._gmm_kl_weights["c"]
    std_var = c._standard_kl_weights["c"]
    _compile(c, gmm_kl_weights={"c": 4.0}, standard_kl_weights={"c": 0.3})
    assert c._gmm_kl_weights["c"] is gmm_var
    assert c._standard_kl_weights["c"] is std_var
    assert float(gmm_var) == 4.0
    assert float(std_var) == pytest.approx(0.3)
    assert c.loss[GMM].weight.scale == 4.0


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
    assert list(child.loss) == [GMM, key, NLL_RNA, NLL_ATAC]
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


@pytest.mark.xfail(
    strict=True,
    reason="BUG: compile() pops entries from the caller's loss_weights dict; "
    "recompiling with the same dict silently resets data weights to 1.0",
)
def test_compile_keeps_caller_loss_weights(fakes):
    c = make_component(name="c")
    lw = {NLL_RNA: 2.0}
    _compile(c, loss_weights=lw)
    _compile(c, loss_weights=lw)
    assert lw == {NLL_RNA: 2.0}
    assert float(c._data_loss_weights["c"]["rna"]) == 2.0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _gmm/_standard_kl_weights are standalone tf.Variables never "
    "passed to the KL losses; changing them has no effect",
)
def test_kl_weight_index_is_tied_to_the_loss_weight(fakes):
    """compile() says weights are backed by tf.Variables 'so callbacks can
    adjust them at runtime'; Model.compile indexes loss.weight itself."""
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 1.0}, gmm_kl_weights={"c": 1.0})
    c._gmm_kl_weights["c"].assign(0.25)
    c._standard_kl_weights["c"].assign(0.5)
    assert c.loss[GMM].weight.scale == pytest.approx(0.25)
    assert c.loss[STD].weight.scale == pytest.approx(0.5)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _data_loss_weights Variable is snapshotted by "
    "NegativeLogDataLikelihood(float(weight)); assigning it is a no-op",
)
def test_data_weight_index_is_tied_to_the_loss_weight(fakes):
    c = make_component(name="c")
    _compile(c)
    c._data_loss_weights["c"]["rna"].assign(0.0)
    y = tf.ones((4, 6))
    assert float(c.loss[NLL_RNA](y, y)) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: docstring says 'A weight > 0 creates the loss' but a 0 "
    "weight still creates it (and its metric becomes 0/0 = NaN)",
)
def test_zero_weight_does_not_create_standard_kl(fakes):
    c = make_component(name="c")
    _compile(c, standard_kl_weights={"c": 0.0}, gmm_kl_weights={"c": 1.0})
    assert STD not in c.loss
