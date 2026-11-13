"""Unit tests for cavachon.callbacks.verbose_callback.VerboseCallback.

Mocked: the model (SimpleNamespace with ``components`` and the
``_*_weights`` dicts), components (SimpleNamespace), progressive scalers
(SimpleNamespace of tf.Variables) and ``time`` (patched at
``cavachon.callbacks.verbose_callback.time``). Output is captured with
capsys and compared exactly.
"""

from types import SimpleNamespace

import pytest
import tensorflow as tf

import cavachon.callbacks.verbose_callback as vc
from cavachon.callbacks.verbose_callback import VerboseCallback

C = VerboseCallback._COLORS
RED, BOLD, RESET = VerboseCallback._RED, VerboseCallback._BOLD, VerboseCallback._RESET


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 100.0}
    monkeypatch.setattr(vc, "time", SimpleNamespace(time=lambda: t["now"]))
    return t


def _scaler(cur, tot):
    return SimpleNamespace(
        current_iteration=tf.Variable(float(cur)),
        total_iterations=tf.Variable(float(tot)),
    )


def _comp(trainable, scaler=None):
    return SimpleNamespace(
        trainable=trainable,
        hierarchical_encoder=SimpleNamespace(progressive_scaler=scaler),
    )


def _model(components, **weights):
    return SimpleNamespace(components=components, **weights)


# --------------------------------------------------------------------------
# prefix matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, expected",
    [
        ("ATAC_DIST_gmm_kl", "ATAC_DIST"),
        ("ATAC_gmm_kl", "ATAC"),
        ("ATACX_gmm_kl", None),  # needs the "_" separator
        ("val_ATAC_gmm_kl", None),
        ("loss", None),
    ],
)
def test_match_component_longest_prefix(key, expected):
    cb = VerboseCallback(labels=[], component_order=["ATAC", "ATAC_DIST"])
    assert cb._match_component(key) == expected


def test_strip_component_prefix():
    cb = VerboseCallback(labels=[], component_order=["ATAC", "ATAC_DIST"])
    assert cb._strip_component_prefix("ATAC_DIST_rna_nll") == "rna_nll"
    assert cb._strip_component_prefix("ATAC_rna_nll") == "rna_nll"
    assert cb._strip_component_prefix("other_metric") == "other_metric"


def test_mutable_defaults_not_shared():
    a = VerboseCallback(labels=[])
    b = VerboseCallback(labels=[])
    a.component_order.append("x")
    a.loss_prefixes["x"] = 0
    assert b.component_order == [] and b.loss_prefixes == {}


# --------------------------------------------------------------------------
# on_epoch_begin banner
# --------------------------------------------------------------------------


def test_epoch_begin_banner_full(capsys, clock):
    cb = VerboseCallback(
        labels=[("parent", 0), ("child", 6)],
        phase="Parent Annealing",
        phase_epochs=20,
        cumulative_offset=40,
        cumulative_total=100,
        phase_number=3,
    )
    cb.on_epoch_begin(4)
    label = f"{C[0]}{BOLD}parent{RESET} → {C[1]}{BOLD}child{RESET}"
    expected = (
        f"{RED}Phase 3. Parent Annealing{RESET} [{label}]"
        f"{RED} Phase Epoch: 5/20 (Total Epoch: 45/100){RESET}\n"
    )
    assert capsys.readouterr().out == expected
    assert cb._epoch_start == 100.0


def test_epoch_begin_banner_without_epoch_counts(capsys, clock):
    cb = VerboseCallback(labels=[("rna", 2)])
    cb.on_epoch_begin(0)
    assert capsys.readouterr().out == (
        f"{RED}Phase 1. Regular Training{RESET} "
        f"[{C[2]}{BOLD}rna{RESET}]{RED}{RESET}\n"
    )


# --------------------------------------------------------------------------
# on_epoch_end
# --------------------------------------------------------------------------


def test_epoch_end_empty_logs_prints_nothing(capsys, clock):
    cb = VerboseCallback(labels=[], component_order=["a"])
    cb.set_model(_model({"a": _comp(True)}))
    cb.on_epoch_begin(0)
    capsys.readouterr()
    cb.on_epoch_end(0, {})
    cb.on_epoch_end(0, None)
    assert capsys.readouterr().out == ""


def test_epoch_end_training_and_frozen_without_debug(capsys, clock):
    cb = VerboseCallback(labels=[], component_order=["a", "b", "c"], debug=False)
    cb.set_model(_model({"a": _comp(True), "b": _comp(False), "c": _comp(True)}))
    cb.on_epoch_begin(0)
    capsys.readouterr()
    clock["now"] = 112.34
    logs = {
        "loss": 10.0,
        "a_gmm_kl": 1.23456,
        "a_rna_nll": 2,
        "b_gmm_kl": 0.5,
        "a_text": "skip-me",  # non-numeric -> ignored
        "orphan": 7.0,  # no component -> ignored
    }
    cb.on_epoch_end(0, logs)
    assert capsys.readouterr().out == (
        "→ loss=10.000\n"
        f"→ {C[0]}{BOLD}a (Training): gmm_kl=1.235  rna_nll=2.000{RESET}\n"
        "→ b (Frozen): gmm_kl=0.500\n"
        "→ 12.3s\n"
    )


def test_epoch_end_elapsed_zero_without_epoch_begin(capsys):
    cb = VerboseCallback(labels=[], component_order=[])
    cb.set_model(_model({}))
    cb.on_epoch_end(0, {"loss": 1.0})
    assert capsys.readouterr().out == "→ loss=1.000\n→ 0.0s\n"


def test_epoch_end_debug_lines(capsys, clock):
    cb = VerboseCallback(labels=[], component_order=["a", "b"], debug=True)
    model = _model(
        {"a": _comp(True, _scaler(5, 10)), "b": _comp(False, None)},
        _gmm_kl_weights={"a": tf.constant(0.5)},
        _standard_kl_weights={"a": tf.constant(0.25)},
        _z_hat_gmm_kl_weights={"a": tf.constant(0.75)},
        _data_loss_weights={"a": {"rna": tf.constant(1.0), "atac": tf.constant(0.5)}},
    )
    cb.set_model(model)
    cb.on_epoch_begin(0)
    capsys.readouterr()
    cb.on_epoch_end(0, {"a_kl": 1.0, "b_kl": 2.0})
    out = capsys.readouterr().out.splitlines()
    indent_a = " " * (len("→ ") + len("a") + len("(Training): ") + 1)
    indent_b = " " * (len("→ ") + len("b") + len("(Frozen): ") + 1)
    assert out == [
        f"→ {C[0]}{BOLD}a (Training): kl=1.000{RESET}",
        f"{C[0]}{indent_a}GMM weight=0.50  KL weight=0.25  z_hat GMM weight=0.75"
        f"  data weight=1.50  α²=0.25{RESET}",
        "→ b (Frozen): kl=2.000",
        f"{indent_b}GMM weight=N/A  KL weight=N/A  data weight=0.00  α²=N/A",
        "→ 0.0s",
    ]


def test_debug_trainable_component_without_metrics(capsys, clock):
    cb = VerboseCallback(labels=[], component_order=["a", "b"], debug=True)
    cb.set_model(_model({"a": _comp(True, _scaler(3, 0)), "b": _comp(False)}))
    cb.on_epoch_end(0, {"loss": 2.0})
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "→ loss=2.000"
    assert out[1] == f"→ {C[0]}{BOLD}a (Training): (no metrics){RESET}"
    # total_iterations == 0 must not divide by zero: alpha clamps to 1.
    assert out[2].endswith(f"α²=1.00{RESET}")
    assert "GMM weight=N/A" in out[2] and "data weight=N/A" in out[2]
    # frozen component without metrics is skipped entirely
    assert out[3:] == ["→ 0.0s"]


def test_model_without_components_reports_frozen(capsys, clock):
    cb = VerboseCallback(labels=[], component_order=["a"], debug=True)
    cb.set_model(SimpleNamespace())
    cb.on_epoch_end(0, {"a_kl": 3.0})
    assert capsys.readouterr().out == "→ a (Frozen): kl=3.000\n→ 0.0s\n"


def test_color_cycles_by_component_order_index(capsys, clock):
    order = [f"c{i}" for i in range(7)]
    cb = VerboseCallback(labels=[], component_order=order, debug=False)
    cb.set_model(_model({n: _comp(True) for n in order}))
    cb.on_epoch_end(0, {"c6_kl": 1.0})
    assert capsys.readouterr().out.splitlines()[0] == (
        f"→ {C[6 % len(C)]}{BOLD}c6 (Training): kl=1.000{RESET}"
    )
