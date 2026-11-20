"""Unit tests for cavachon.callbacks.annealing_callback.AnnealingCallback.

Mocked: ProgressiveScaler (patched at
``cavachon.callbacks.annealing_callback.ProgressiveScaler``), the model
(FakeCavachonModel), components (SimpleNamespace) and the k-means
initializer (MagicMock). Loss weights are real tf.Variables.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from cavachon.callbacks.annealing_callback import AnnealingCallback
from cavachon.environment.constants import Constants

STD = Constants.MODEL_LOSS_STANDARD_KL_POSTFIX
GMM = Constants.MODEL_LOSS_GMM_KL_POSTFIX
ZHAT_KL = Constants.MODEL_LOSS_Z_HAT_GMM_KL_POSTFIX
DATA = Constants.MODEL_LOSS_DATA_POSTFIX


def _weights(model):
    return {k: float(v.weight.numpy()) for k, v in model.loss.items()}


@pytest.fixture
def child_losses(loss_factory):
    return {
        f"child1_{STD}": loss_factory(9.0),
        f"child1_{GMM}": loss_factory(9.0),
        f"child1_{ZHAT_KL}": loss_factory(9.0),
        f"child1_RNA_{DATA}": loss_factory(9.0),
        f"parent_{GMM}": loss_factory(9.0),
    }


# --------------------------------------------------------------------------
# on_epoch_begin: weight assignment
# --------------------------------------------------------------------------


def test_longest_pattern_wins_and_unmatched_untouched(make_model, child_losses):
    model = make_model(loss=child_losses)
    cb = AnnealingCallback(schedule=lambda e: {"child1": 0.5, f"child1_{STD}": 0.2})
    cb.set_model(model)
    cb.on_epoch_begin(0)
    w = _weights(model)
    assert w[f"child1_{STD}"] == pytest.approx(0.2)
    assert w[f"child1_{GMM}"] == pytest.approx(0.5)
    assert w[f"child1_RNA_{DATA}"] == pytest.approx(0.5)
    # z_hat KL is not switched on by the component-wide pattern.
    assert w[f"child1_{ZHAT_KL}"] == pytest.approx(9.0)
    assert w[f"parent_{GMM}"] == pytest.approx(9.0)


def test_z_hat_kl_only_matched_by_pattern_containing_its_postfix(
    make_model, child_losses
):
    model = make_model(loss=child_losses)
    cb = AnnealingCallback(schedule=lambda e: {GMM: 0.3, f"child1_{ZHAT_KL}": 0.7})
    cb.set_model(model)
    cb.on_epoch_begin(0)
    w = _weights(model)
    # "gmm_kl_divergence" is a substring of the z_hat postfix, but must not
    # match the z_hat loss.
    assert w[f"child1_{ZHAT_KL}"] == pytest.approx(0.7)
    assert w[f"child1_{GMM}"] == pytest.approx(0.3)
    assert w[f"parent_{GMM}"] == pytest.approx(0.3)


def test_schedule_receives_epoch_and_int_weight_cast_to_float32(
    make_model, loss_factory
):
    model = make_model(loss={f"c_{GMM}": loss_factory(0.0)})
    schedule = MagicMock(return_value={"c": 1})
    cb = AnnealingCallback(schedule=schedule)
    cb.set_model(model)
    cb.on_epoch_begin(7)
    schedule.assert_called_once_with(7)
    var = model.loss[f"c_{GMM}"].weight
    assert var.dtype == tf.float32
    assert float(var.numpy()) == 1.0


def test_losses_without_assignable_weight_are_skipped(make_model, loss_factory):
    plain_fn = lambda y, p: 0.0  # noqa: E731 - no .weight at all
    float_weight = SimpleNamespace(weight=0.25)  # .weight without .assign
    model = make_model(
        loss={
            "c_fn": plain_fn,
            "c_float": float_weight,
            f"c_{GMM}": loss_factory(0.0),
        }
    )
    cb = AnnealingCallback(schedule=lambda e: {"c": 0.9})
    cb.set_model(model)
    cb.on_epoch_begin(0)
    assert float_weight.weight == 0.25
    assert float(model.loss[f"c_{GMM}"].weight.numpy()) == pytest.approx(0.9)


def test_empty_schedule_changes_nothing(make_model, child_losses):
    model = make_model(loss=child_losses)
    cb = AnnealingCallback(schedule=lambda e: {})
    cb.set_model(model)
    cb.on_epoch_begin(3)
    assert set(_weights(model).values()) == {9.0}


def test_schedule_evaluated_per_epoch(make_model, loss_factory):
    model = make_model(loss={f"c_{STD}": loss_factory(0.0)})
    cb = AnnealingCallback(schedule=lambda e: {f"c_{STD}": e / 10})
    cb.set_model(model)
    seen = []
    for epoch in range(4):
        cb.on_epoch_begin(epoch)
        seen.append(float(model.loss[f"c_{STD}"].weight.numpy()))
    np.testing.assert_allclose(seen, [0.0, 0.1, 0.2, 0.3], rtol=1e-6)


# --------------------------------------------------------------------------
# on_epoch_begin: k-means initialisation
# --------------------------------------------------------------------------


def _kmeans_callback(make_model, make_comp, kmeans_init):
    comps = {"a": make_comp(trainable=True), "b": make_comp(trainable=False)}
    model = make_model(components=comps)
    cb = AnnealingCallback(
        schedule=lambda e: {},
        kmeans_epoch=2,
        kmeans_init=kmeans_init,
        component_name="a",
    )
    cb.set_model(model)
    return cb, model, comps


def test_kmeans_runs_once_at_kmeans_epoch(make_model, make_comp):
    init = MagicMock(name="kmeans_init")
    cb, _, _ = _kmeans_callback(make_model, make_comp, init)
    for epoch in (0, 1):
        cb.on_epoch_begin(epoch)
    init.assert_not_called()
    cb.on_epoch_begin(2)
    init.assert_called_once_with()
    # A second visit of the same epoch (e.g. a re-fit) does not re-run it.
    cb.on_epoch_begin(2)
    assert init.call_count == 1
    assert cb._kmeans_done is True


def test_kmeans_leaves_trainable_flags_to_the_initializer(make_model, make_comp):
    # the initializer freezes the model and restores every flag itself
    cb, model, comps = _kmeans_callback(make_model, make_comp, MagicMock())
    cb.on_epoch_begin(2)
    assert comps["a"].trainable is True
    assert comps["b"].trainable is False


def test_kmeans_disabled_when_epoch_none(make_model, make_comp):
    init = MagicMock(name="kmeans_init")
    model = make_model(components={"a": make_comp()})
    cb = AnnealingCallback(schedule=lambda e: {}, kmeans_epoch=None, kmeans_init=init)
    cb.set_model(model)
    for epoch in range(5):
        cb.on_epoch_begin(epoch)
    init.assert_not_called()


def test_kmeans_epoch_without_initializer_is_a_no_op(make_model, make_comp):
    model = make_model(components={"a": make_comp()})
    cb = AnnealingCallback(schedule=lambda e: {}, kmeans_epoch=0)
    cb.set_model(model)
    cb.on_epoch_begin(0)
    assert cb._kmeans_done is True


def test_component_names_are_copied():
    names = ["a"]
    cb = AnnealingCallback(schedule=lambda e: {}, component_names=names)
    names.append("b")
    assert cb.component_names == ["a"]
    assert AnnealingCallback(schedule=lambda e: {}).component_names == []


# --------------------------------------------------------------------------
# on_batch_end: progressive scaler increments
# --------------------------------------------------------------------------


def test_on_batch_end_increments_all_active_scalers(
    make_model, make_comp, fake_scaler_cls
):
    enc = fake_scaler_cls(total_iterations=100)
    std_c = fake_scaler_cls(total_iterations=50)
    gmm_c = fake_scaler_cls(total_iterations=50)
    gmm_d_pinned = fake_scaler_cls(total_iterations=1)  # pinned -> skipped
    data_rna = fake_scaler_cls(total_iterations=20)
    data_atac_pinned = fake_scaler_cls(total_iterations=1)
    not_a_scaler = MagicMock(total_iterations=100.0)
    model = make_model(
        components={"c": make_comp(scaler=enc)},
        _standard_kl_weights={"c": std_c, "d": not_a_scaler},
        _gmm_kl_weights={"c": gmm_c, "d": gmm_d_pinned},
        _data_loss_weights={"c": {"RNA": data_rna, "ATAC": data_atac_pinned}},
    )
    cb = AnnealingCallback(
        schedule=lambda e: {}, component_name="c", component_names=["c", "d"]
    )
    cb.set_model(model)
    cb.on_batch_end(0)
    cb.on_batch_end(1)
    assert enc.increment_calls == 2
    assert std_c.increment_calls == 2
    assert gmm_c.increment_calls == 2
    assert data_rna.increment_calls == 2
    assert gmm_d_pinned.increment_calls == 0
    assert data_atac_pinned.increment_calls == 0
    not_a_scaler.increment.assert_not_called()


def test_on_train_batch_end_routes_to_on_batch_end(
    make_model, make_comp, fake_scaler_cls
):
    enc = fake_scaler_cls(total_iterations=5)
    model = make_model(components={"c": make_comp(scaler=enc)})
    cb = AnnealingCallback(schedule=lambda e: {}, component_name="c")
    cb.set_model(model)
    cb.on_train_batch_end(0)
    assert enc.increment_calls == 1


def test_on_batch_end_tolerates_missing_component(
    make_model, make_comp, fake_scaler_cls
):
    std_c = fake_scaler_cls(total_iterations=5)
    model = make_model(components={}, _standard_kl_weights={"c": std_c})
    cb = AnnealingCallback(
        schedule=lambda e: {}, component_name="missing", component_names=["c"]
    )
    cb.set_model(model)
    cb.on_batch_end(0)  # missing component -> swallowed
    assert std_c.increment_calls == 1

    enc = fake_scaler_cls(total_iterations=5)
    model2 = make_model(components={"c": make_comp(scaler=enc)})
    cb2 = AnnealingCallback(schedule=lambda e: {}, component_name="c")
    cb2.set_model(model2)
    cb2.on_batch_end(0)  # no component_names -> only alpha advances
    assert enc.increment_calls == 1
