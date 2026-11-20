"""Model.compile (cavachon/model/model.py).

Mocked: StandardKLDivergence / GMMKLDivergence / NegativeLogDataLikelihood
(-> FakeStandardKL / FakeGMMKL / FakeNLL, each holding a FakeScaler weight),
GMMDensityLoss (-> FakeGMMDensity), ProgressiveScaler (-> FakeScaler),
Component (-> FakeComponent), GeneralUtils. ComponentConfig -> FakeConfig.
"""

import pytest
import tensorflow as tf

from test.model.conftest import (
    FakeConfig,
    FakeGMMDensity,
    FakeGMMKL,
    FakeNLL,
    FakeScaler,
    FakeStandardKL,
    PlainLoss,
)

STD = "standard_kl_divergence"
GMM = "gmm_kl_divergence"
NLL = "negative_log_data_likelihood"
ZD = "z_hat_gmm_density"
ZKL = "z_hat_gmm_kl_divergence"


@pytest.fixture
def two_component_model(make_model):
    a = FakeConfig("a", ["rna"], distribution_names={"rna": "NegativeBinomial"})
    b = FakeConfig(
        "b",
        ["rna", "atac"],
        n_vars={"rna": 2, "atac": 3},
        distribution_names={"rna": "NegativeBinomial", "atac": "Bernoulli"},
        conditioned_on_z=["a"],
    )
    return make_model(a, b)


class TestKLLosses:
    def test_default_is_gmm_kl_weight_one(self, two_component_model):
        m = two_component_model
        m.compile()
        assert set(m.loss) == {
            f"a_{GMM}",
            f"b_{GMM}",
            f"a_rna_{NLL}",
            f"b_rna_{NLL}",
            f"b_atac_{NLL}",
        }
        for c in ("a", "b"):
            loss = m.loss[f"{c}_{GMM}"]
            assert isinstance(loss, FakeGMMKL)
            assert loss.name == f"{c}_{GMM}"
            assert loss.init_weight == 1.0
            assert m._gmm_kl_weights[c] is loss.weight
        assert FakeStandardKL.instances == []
        assert m._standard_kl_weights == {}

    def test_standard_and_gmm_weights(self, two_component_model):
        m = two_component_model
        m.compile(standard_kl_weights={"a": 0.3, "b": 0.7}, gmm_kl_weights={"a": 2.0})
        assert m.loss[f"a_{STD}"].init_weight == 0.3
        assert m.loss[f"b_{STD}"].init_weight == 0.7
        assert m.loss[f"a_{GMM}"].init_weight == 2.0
        # b only in standard dict -> no GMM KL for b
        assert f"b_{GMM}" not in m.loss
        assert set(m._standard_kl_weights) == {"a", "b"}
        assert set(m._gmm_kl_weights) == {"a"}
        assert m._standard_kl_weights["b"] is m.loss[f"b_{STD}"].weight
        assert isinstance(m._standard_kl_weights["a"], FakeScaler)
        assert m.loss[f"a_{STD}"].name == f"a_{STD}"

    def test_component_only_in_gmm_dict(self, two_component_model):
        m = two_component_model
        m.compile(gmm_kl_weights={"a": 0.5})
        assert m.loss[f"a_{GMM}"].init_weight == 0.5
        # b absent from both -> default GMM 1.0
        assert m.loss[f"b_{GMM}"].init_weight == 1.0
        assert f"a_{STD}" not in m.loss

    def test_scheduler_drives_weights_through_index(self, two_component_model):
        m = two_component_model
        m.compile(standard_kl_weights={"a": 1.0}, gmm_kl_weights={"a": 1.0})
        m._standard_kl_weights["a"].activate(50)
        m._gmm_kl_weights["a"].pin_to(0.25)
        m._data_loss_weights["b"]["atac"].pin_to(0.5)
        assert m.loss[f"a_{STD}"].weight.activations == [50.0]
        assert m.loss[f"a_{GMM}"].weight.pins == [0.25]
        assert m.loss[f"b_atac_{NLL}"].weight.pins == [0.5]

    def test_zero_weight_creates_the_loss_switched_off(self, two_component_model):
        # regression: a weight of 0 created the loss with scale 0, so it
        # could never be switched on and its reported value was 0 / 0.
        # It is now created with scale 1 and pinned to 0 (frozen parents
        # are still logged without affecting training).
        m = two_component_model
        m.compile(standard_kl_weights={"a": 0.0}, gmm_kl_weights={"a": 1.0, "b": 0.0})
        for key in (f"a_{STD}", f"b_{GMM}"):
            assert m.loss[key].init_weight == 1.0
            assert m.loss[key].weight.pins == [0.0]
        assert m.loss[f"a_{GMM}"].init_weight == 1.0
        assert m.loss[f"a_{GMM}"].weight.pins == []

    def test_nonzero_weight_is_the_scale(self, two_component_model):
        m = two_component_model
        m.compile(standard_kl_weights={"a": 2.5}, loss_weights={f"b_atac_{NLL}": 0.0})
        assert m.loss[f"a_{STD}"].init_weight == 2.5
        assert m.loss[f"a_{STD}"].weight.pins == []
        nll = m.loss[f"b_atac_{NLL}"]
        assert nll.init_weight == 1.0 and nll.weight.pins == [0.0]


class TestDataLosses:
    def test_nll_per_component_and_modality(self, two_component_model):
        m = two_component_model
        m.compile()
        expected = {
            ("a", "rna"): "NegativeBinomial",
            ("b", "rna"): "NegativeBinomial",
            ("b", "atac"): "Bernoulli",
        }
        for (c, mod), dist in expected.items():
            loss = m.loss[f"{c}_{mod}_{NLL}"]
            assert isinstance(loss, FakeNLL)
            assert loss.distribution_name == dist
            assert loss.name == f"{c}_{mod}_{NLL}"
            assert loss.init_weight == 1.0
            assert m._data_loss_weights[c][mod] is loss.weight
        assert set(m._data_loss_weights) == {"a", "b"}
        assert set(m._data_loss_weights["b"]) == {"rna", "atac"}

    def test_loss_weights_kwarg_sets_nll_scale_and_is_not_forwarded(
        self, two_component_model, monkeypatch
    ):
        m = two_component_model
        seen = {}
        orig = tf.keras.Model.compile

        def spy(self, **kwargs):
            seen.update(kwargs)
            return orig(self, **kwargs)

        monkeypatch.setattr(tf.keras.Model, "compile", spy)
        m.compile(loss_weights={f"b_atac_{NLL}": 3.0})
        assert m.loss[f"b_atac_{NLL}"].init_weight == 3.0
        assert m.loss[f"a_rna_{NLL}"].init_weight == 1.0
        assert "loss_weights" not in seen
        assert seen["loss"] is not None

    def test_loss_weights_dict_of_caller_not_mutated(self, two_component_model):
        # regression: compile() popped entries from the caller's dict
        lw = {f"b_atac_{NLL}": 3.0}
        two_component_model.compile(loss_weights=lw)
        assert lw == {f"b_atac_{NLL}": 3.0}
        assert two_component_model.loss[f"b_atac_{NLL}"].init_weight == 3.0

    def test_unknown_loss_weights_warn(self, two_component_model):
        typo = f"b_atak_{NLL}"
        with pytest.warns(RuntimeWarning, match=typo):
            two_component_model.compile(loss_weights={typo: 3.0})


class TestZHatLosses:
    def _model(self, make_model, **kw):
        a = FakeConfig("a", ["rna"], learn_z_hat_priors=True, **kw)
        b = FakeConfig("b", ["rna"], learn_z_hat_priors=False)
        return make_model(a, b)

    def test_density_and_kl_created_for_learned_priors(self, make_model):
        m = self._model(make_model)
        m.compile()
        dens = m.loss[f"a_{ZD}"]
        assert isinstance(dens, FakeGMMDensity)
        assert dens.name == f"a_{ZD}"
        assert isinstance(dens.weight, tf.Variable)
        assert dens.weight is m._z_hat_density_weights["a"]
        assert float(dens.weight) == 1.0
        assert dens.weight.trainable is False
        assert f"b_{ZD}" not in m.loss and "b" not in m._z_hat_density_weights

        zkl = m.loss[f"a_{ZKL}"]
        assert isinstance(zkl, FakeGMMKL)
        assert zkl.name == f"a_{ZKL}"
        # z_hat_kl_weight None -> 1.0 target scale, pinned to 0 at compile
        assert zkl.init_weight == 1.0
        assert zkl.weight.pins == [0.0]
        assert m._z_hat_gmm_kl_weights == {"a": zkl.weight}

    def test_custom_z_hat_kl_weight(self, make_model):
        m = self._model(make_model, z_hat_kl_weight=0.2)
        m.compile()
        assert m.loss[f"a_{ZKL}"].init_weight == 0.2
        assert m.loss[f"a_{ZKL}"].weight.pins == [0.0]

    def test_zero_z_hat_kl_weight_creates_no_kl(self, make_model):
        m = self._model(make_model, z_hat_kl_weight=0.0)
        m.compile()
        assert f"a_{ZKL}" not in m.loss
        assert m._z_hat_gmm_kl_weights == {}
        # density loss still there
        assert f"a_{ZD}" in m.loss

    def test_no_z_hat_losses_without_parameterizer(self, make_model):
        m = make_model(
            FakeConfig(
                "a", learn_z_hat_priors=True, has_z_hat_prior_parameterizer=False
            )
        )
        m.compile()
        assert f"a_{ZD}" not in m.loss and f"a_{ZKL}" not in m.loss
        assert m._z_hat_density_weights == {} and m._z_hat_gmm_kl_weights == {}


class TestRecompile:
    def test_recompile_reuses_losses_and_variables(self, make_model):
        m = make_model(FakeConfig("a", learn_z_hat_priors=True))
        m.compile()
        first = dict(m.loss)
        var = m._z_hat_density_weights["a"]
        zkl_weight = m._z_hat_gmm_kl_weights["a"]
        zkl_weight.pin_to(0.7)
        m.compile(gmm_kl_weights={"a": 5.0})
        for k, v in first.items():
            assert m.loss[k] is v
        assert m._z_hat_density_weights["a"] is var
        # z_hat KL is not re-pinned to 0 on recompile
        assert zkl_weight.pins == [0.0, 0.7]
        # existing GMM loss keeps its original scale
        assert m.loss[f"a_{GMM}"].init_weight == 1.0
        # existing losses are not rebuilt (and discarded) on recompile
        assert len(FakeGMMKL.instances) == 2  # a_gmm + a_z_hat_kl
        assert len(FakeNLL.instances) == 1

    def test_recompile_adds_new_loss_types(self, make_model):
        m = make_model(FakeConfig("a"))
        m.compile()
        m.compile(standard_kl_weights={"a": 1.0}, gmm_kl_weights={"a": 1.0})
        assert f"a_{STD}" in m.loss and f"a_{GMM}" in m.loss


class TestCustomLossAndMetrics:
    def test_custom_loss_warns_and_is_used(self, make_model):
        m = make_model(FakeConfig("a"))
        custom = {"a_rna_negative_log_data_likelihood": PlainLoss()}
        with pytest.warns(RuntimeWarning, match="custom losses"):
            m.compile(loss=custom)
        assert dict(m.loss) == custom
        assert FakeNLL.instances == [] and FakeGMMKL.instances == []
        assert m._gmm_kl_weights == {} and m._standard_kl_weights == {}

    def test_metrics_are_dropped_with_warning(self, make_model, monkeypatch):
        m = make_model(FakeConfig("a"))
        seen = {}
        orig = tf.keras.Model.compile

        def spy(self, **kwargs):
            seen.update(kwargs)
            return orig(self, **kwargs)

        monkeypatch.setattr(tf.keras.Model, "compile", spy)
        with pytest.warns(RuntimeWarning, match="metrics"):
            m.compile(metrics=["mse"])
        assert "metrics" not in seen

    def test_optimizer_forwarded(self, make_model):
        m = make_model(FakeConfig("a"))
        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        m.compile(optimizer=opt)
        assert m.optimizer is opt
