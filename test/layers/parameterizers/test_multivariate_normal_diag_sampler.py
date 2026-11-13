import pytest
import numpy as np
import tensorflow as tf

from cavachon.layers.parameterizers import MultivariateNormalDiagSampler as Sampler

LOC = np.array([[1.0, -2.0], [0.5, 3.0]], np.float32)
SCALE = np.array([[0.1, 2.0], [1.0, 0.5]], np.float32)
PARAMS = tf.constant(np.concatenate([LOC, SCALE], -1))


def test_inference_returns_loc():
    np.testing.assert_array_equal(Sampler()(PARAMS).numpy(), LOC)
    np.testing.assert_array_equal(Sampler()(PARAMS, training=False).numpy(), LOC)


def test_training_samples_with_correct_moments():
    s = Sampler()
    big = tf.tile(PARAMS[:1], (20000, 1))
    out = s(big, training=True).numpy()
    np.testing.assert_allclose(out.mean(0), LOC[0], atol=0.05)
    np.testing.assert_allclose(out.std(0), SCALE[0], rtol=0.05)


def test_non_stochastic_returns_loc_in_training():
    np.testing.assert_allclose(Sampler(stochastic=False)(PARAMS, training=True).numpy(), LOC)


def test_stochastic_variable_toggle():
    s = Sampler()
    assert not s.stochastic.trainable
    s.stochastic.assign(0.0)
    np.testing.assert_allclose(s(PARAMS, training=True).numpy(), LOC)


def test_frozen_layer_returns_loc():
    s = Sampler()
    s.trainable = False
    np.testing.assert_array_equal(s(PARAMS, training=True).numpy(), LOC)


def test_reparameterized_gradient():
    p = tf.Variable(PARAMS)
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(Sampler()(p, training=True))
    g = tape.gradient(y, p).numpy()
    np.testing.assert_allclose(g[:, :2], 1.0)
    assert np.abs(g[:, 2:]).sum() > 0


def test_zero_scale_gives_loc():
    params = tf.constant(np.concatenate([LOC, np.zeros_like(SCALE)], -1))
    np.testing.assert_allclose(Sampler()(params, training=True).numpy(), LOC)


def test_get_config_contains_init_args():
    kwargs = dict(stochastic=False, name="smp")
    cfg = Sampler(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


def test_from_config_round_trip():
    # regression: __init__ lacked **kwargs, so from_config (trainable/dtype) raised TypeError
    clone = Sampler.from_config(Sampler(**dict(stochastic=False, name="smp")).get_config())
    assert float(clone.stochastic) == 0.0 and clone.name == "smp"
