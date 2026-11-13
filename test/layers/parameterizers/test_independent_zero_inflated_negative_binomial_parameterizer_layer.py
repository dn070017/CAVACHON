import numpy as np
import pytest
import tensorflow as tf

from cavachon.layers.parameterizers import (
    IndependentZeroInflatedNegativeBinomialParameterizerLayer as ZINB,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@pytest.mark.parametrize("shared", [False, True])
def test_output_shape_and_ranges(shared):
    layer = ZINB(event_dims=4, use_shared_dispersion=shared)
    out = layer(tf.random.normal((3, 6), seed=1)).numpy()
    assert out.shape == (3, 12)
    means, disp = out[:, 4:8], out[:, 8:]
    np.testing.assert_allclose(means.sum(-1), 1.0, rtol=1e-5)
    assert (means > 0).all()
    assert (disp >= 0.1).all() and (disp <= 1.0).all()


def test_numeric_reference_independent():
    layer = ZINB(event_dims=3)
    x = tf.random.normal((2, 5), seed=2)
    out = layer(x).numpy()
    xn = x.numpy()
    logits = xn @ layer.logits_weight.numpy() + layer.logits_bias.numpy()
    means = _softmax(xn @ layer.mean_weight.numpy() + layer.mean_bias.numpy())
    disp = np.maximum(_sigmoid(xn @ layer.dispersion_weight.numpy() + layer.dispersion_bias.numpy()), 0.1)
    np.testing.assert_allclose(out, np.concatenate([logits, means, disp], -1), rtol=1e-5, atol=1e-6)


def test_shared_dispersion_independent_of_input():
    layer = ZINB(event_dims=3, use_shared_dispersion=True)
    out = layer(tf.random.normal((4, 5), seed=3)).numpy()
    assert tuple(layer.dispersion_weight.shape) == (1, 3)
    np.testing.assert_allclose(out[:, 6:], np.repeat(out[:1, 6:], 4, axis=0))


def test_dispersion_clamped_to_point_one():
    layer = ZINB(event_dims=2)
    layer.build((None, 2))
    layer.dispersion_bias.assign(-100.0 * np.ones((1, 2), np.float32))
    out = layer(tf.zeros((1, 2))).numpy()
    np.testing.assert_allclose(out[:, 4:], 0.1)


def test_softmax_stable_for_large_inputs():
    layer = ZINB(event_dims=3)
    out = layer(tf.fill((2, 4), 1e4)).numpy()
    assert np.isfinite(out[:, 3:]).all()


def test_get_config_contains_init_args():
    kwargs = dict(event_dims=3, use_shared_dispersion=True)
    cfg = ZINB(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


@pytest.mark.xfail(
    strict=True,
    reason="BUG: independent_zero_inflated_negative_binomial_parameterizer_layer.py:12 __init__ takes no **kwargs, so Keras from_config (which passes "
    "trainable/dtype from get_config) raises TypeError; breaks clone_model / model save-load",
)
def test_from_config_round_trip():
    clone = ZINB.from_config(ZINB(**dict(event_dims=3, use_shared_dispersion=True)).get_config())
    assert clone.event_dims == 3 and clone.use_shared_dispersion is True
