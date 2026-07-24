"""QA tests for Model.encode(), Model.hierarchical_encode(), and Model.decode() public APIs.

Run with:
    .pixi/envs/default/bin/python test_model_encode_decode.py

All tests print PASS/FAIL and a final summary.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from cavachon.config.component_config import ComponentConfig
from cavachon.model.model import Model

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    _results.append(condition)


def make_model():
    comp_A_config = ComponentConfig(
        name="comp_A",
        modalities=[{"name": "RNA", "distribution_names": "MultivariateNormalDiag", "n_vars": 8}],
        n_vars={"RNA": 8},
        n_vars_batch_effect={"RNA": 2},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=[],
        conditioned_on_z_hat=[],
    )
    comp_B_config = ComponentConfig(
        name="comp_B",
        modalities=[{"name": "ATAC", "distribution_names": "MultivariateNormalDiag", "n_vars": 6}],
        n_vars={"ATAC": 6},
        n_vars_batch_effect={"ATAC": 2},
        n_latent_dims=4,
        n_latent_priors=9,
        n_encoder_layers=1,
        conditioned_on_z=["comp_A"],
        conditioned_on_z_hat=[],
    )
    return Model.make([comp_A_config, comp_B_config])


def make_batch(batch_size=4):
    return {
        "RNA_matrix": tf.constant(np.random.rand(batch_size, 8).astype("float32")),
        "RNA_batch_effect": tf.constant(np.random.rand(batch_size, 2).astype("float32")),
        "ATAC_matrix": tf.constant(np.random.rand(batch_size, 6).astype("float32")),
        "ATAC_batch_effect": tf.constant(np.random.rand(batch_size, 2).astype("float32")),
    }


_model = None


def get_model():
    global _model
    if _model is None:
        _model = make_model()
    return _model


def test_model_encode_happy_path():
    print("\n[Test 1] Model.encode() happy path")
    model = get_model()
    batch = make_batch()

    result = model.encode(batch, training=False)

    check("returns 'z_parameters' key", "z_parameters" in result)
    check("returns 'z' key", "z" in result)
    check("z_parameters keyed by comp_A", "comp_A" in result["z_parameters"])
    check("z_parameters keyed by comp_B", "comp_B" in result["z_parameters"])
    check("z keyed by comp_A", "comp_A" in result["z"])
    check("z keyed by comp_B", "comp_B" in result["z"])
    check("comp_A z shape", result["z"]["comp_A"].shape == (4, 4),
          str(result["z"]["comp_A"].shape))
    check("comp_B z shape", result["z"]["comp_B"].shape == (4, 4),
          str(result["z"]["comp_B"].shape))


def test_model_encode_with_filter():
    print("\n[Test 2] Model.encode() with components filter")
    model = get_model()
    batch = make_batch()

    result = model.encode(batch, components=["comp_A"], training=False)

    check("z has comp_A", "comp_A" in result["z"])
    check("z does not have comp_B", "comp_B" not in result["z"])


def test_model_encode_unknown_component():
    print("\n[Test 3] Model.encode() raises ValueError on unknown component")
    model = get_model()
    batch = make_batch()

    try:
        model.encode(batch, components=["nonexistent"], training=False)
        check("raises ValueError", False, "no exception raised")
    except ValueError as e:
        check("raises ValueError", True)
        check("error mentions unknown component", "nonexistent" in str(e), str(e))


def test_model_hierarchical_encode_happy_path():
    print("\n[Test 4] Model.hierarchical_encode() happy path")
    model = get_model()
    batch = make_batch()

    enc = model.encode(batch, training=False)
    result = model.hierarchical_encode(batch, enc["z"], training=False)

    check("returns 'z_hat' key", "z_hat" in result)
    check("z_hat keyed by comp_A", "comp_A" in result["z_hat"])
    check("z_hat keyed by comp_B", "comp_B" in result["z_hat"])
    check("comp_A z_hat shape", result["z_hat"]["comp_A"].shape == (4, 4),
          str(result["z_hat"]["comp_A"].shape))
    check("comp_B z_hat shape", result["z_hat"]["comp_B"].shape == (4, 4),
          str(result["z_hat"]["comp_B"].shape))


def test_model_hierarchical_encode_strict_missing_z():
    print("\n[Test 5] Model.hierarchical_encode() strict=True raises on missing z")
    model = get_model()
    batch = make_batch()

    try:
        model.hierarchical_encode(batch, {}, strict=True, training=False)
        check("raises ValueError", False, "no exception raised")
    except ValueError as e:
        check("raises ValueError", True, str(e))


def test_model_decode_happy_path():
    print("\n[Test 6] Model.decode() happy path")
    model = get_model()
    batch = make_batch()

    enc = model.encode(batch, training=False)
    hier = model.hierarchical_encode(batch, enc["z"], training=False)
    result = model.decode(batch, hier["z_hat"], training=False)

    check("returns 'x_parameters' key", "x_parameters" in result)
    check("x_parameters has comp_A_RNA_x_parameters",
          "comp_A_RNA_x_parameters" in result["x_parameters"])
    check("x_parameters has comp_B_ATAC_x_parameters",
          "comp_B_ATAC_x_parameters" in result["x_parameters"])
    check("comp_A RNA x_parameters batch dim",
          result["x_parameters"]["comp_A_RNA_x_parameters"].shape[0] == 4,
          str(result["x_parameters"]["comp_A_RNA_x_parameters"].shape))


def test_model_decode_strict_missing_z_hat():
    print("\n[Test 7] Model.decode() strict=True raises on missing z_hat")
    model = get_model()
    batch = make_batch()

    try:
        model.decode(batch, {}, strict=True, training=False)
        check("raises ValueError", False, "no exception raised")
    except ValueError as e:
        check("raises ValueError", True, str(e))


def test_model_decode_strict_false_skips_missing():
    print("\n[Test 8] Model.decode() strict=False skips missing component silently")
    model = get_model()
    batch = make_batch()

    enc = model.encode(batch, training=False)
    hier = model.hierarchical_encode(batch, enc["z"], training=False)
    z_hat_partial = {"comp_A": hier["z_hat"]["comp_A"]}

    result = model.decode(batch, z_hat_partial, strict=False, training=False)

    check("returns x_parameters", "x_parameters" in result)
    check("comp_A present", "comp_A_RNA_x_parameters" in result["x_parameters"])
    check("comp_B absent (skipped)", "comp_B_ATAC_x_parameters" not in result["x_parameters"])


def test_model_full_staged_roundtrip():
    print("\n[Test 9] Full staged round-trip encode → hierarchical_encode → decode")
    model = get_model()
    batch = make_batch(batch_size=6)

    enc = model.encode(batch, training=False)
    hier = model.hierarchical_encode(batch, enc["z"], training=False)
    dec = model.decode(batch, hier["z_hat"], training=False)

    check("enc z comp_A shape", enc["z"]["comp_A"].shape == (6, 4),
          str(enc["z"]["comp_A"].shape))
    check("hier z_hat comp_A shape", hier["z_hat"]["comp_A"].shape == (6, 4),
          str(hier["z_hat"]["comp_A"].shape))
    check("dec RNA x_parameters shape[0]",
          dec["x_parameters"]["comp_A_RNA_x_parameters"].shape[0] == 6,
          str(dec["x_parameters"]["comp_A_RNA_x_parameters"].shape))
    check("dec ATAC x_parameters shape[0]",
          dec["x_parameters"]["comp_B_ATAC_x_parameters"].shape[0] == 6,
          str(dec["x_parameters"]["comp_B_ATAC_x_parameters"].shape))


def test_model_call_backward_compat():
    print("\n[Test 10] Model.__call__() backward compat — output keys unchanged")
    model = get_model()
    batch = make_batch()

    fwd = model(batch, training=False)

    expected_keys = [
        "comp_A_z",
        "comp_A_z_hat",
        "comp_A_z_parameters",
        "comp_A_RNA_x_parameters",
        "comp_B_z",
        "comp_B_z_hat",
        "comp_B_z_parameters",
        "comp_B_ATAC_x_parameters",
    ]
    for key in expected_keys:
        check(f"output has key '{key}'", key in fwd, f"keys={sorted(fwd.keys())}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model encode/decode QA tests")
    print("=" * 60)

    test_model_encode_happy_path()
    test_model_encode_with_filter()
    test_model_encode_unknown_component()
    test_model_hierarchical_encode_happy_path()
    test_model_hierarchical_encode_strict_missing_z()
    test_model_decode_happy_path()
    test_model_decode_strict_missing_z_hat()
    test_model_decode_strict_false_skips_missing()
    test_model_full_staged_roundtrip()
    test_model_call_backward_compat()

    passed = sum(_results)
    total = len(_results)
    print("\n" + "=" * 60)
    if passed == total:
        print(f"\033[92mAll {total} checks passed.\033[0m")
    else:
        print(f"\033[91m{total - passed}/{total} checks FAILED.\033[0m")
    print("=" * 60)
