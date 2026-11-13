import numpy as np
import tensorflow as tf

from cavachon.layers.parameterizers import (
    MultivariateNormalDiagParameterizerLayer as MVN,
)


def _softplus(x):
    return np.logaddexp(0.0, x)


def test_shape_and_numeric_reference():
    layer = MVN(event_dims=3)
    x = tf.random.normal((4, 5), seed=4)
    out = layer(x).numpy()
    assert out.shape == (4, 6)
    xn = x.numpy()
    loc = xn @ layer.loc_weight.numpy() + layer.loc_bias.numpy()
    scale = (
        _softplus(xn @ layer.scale_diag_weight.numpy() + layer.scale_diag_bias.numpy())
        + 1e-7
    )
    np.testing.assert_allclose(out[:, :3], loc, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 3:], scale, rtol=1e-5, atol=1e-6)


def test_scale_strictly_positive_for_very_negative_preactivation():
    layer = MVN(event_dims=2)
    layer.build((None, 1))
    layer.scale_diag_bias.assign(-1e3 * np.ones((1, 2), np.float32))
    out = layer(tf.zeros((1, 1))).numpy()
    assert (out[:, 2:] > 0).all()


def test_softplus_no_overflow():
    layer = MVN(event_dims=2)
    out = layer(tf.fill((1, 3), 1e4)).numpy()
    assert np.isfinite(out).all()


def test_get_config_contains_init_args():
    kwargs = dict(event_dims=7, name="z")
    cfg = MVN(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


def test_from_config_round_trip():
    # regression: __init__ lacked **kwargs, so from_config (trainable/dtype) raised TypeError
    clone = MVN.from_config(MVN(**dict(event_dims=7, name="z")).get_config())
    assert clone.event_dims == 7 and clone.name == "z"
