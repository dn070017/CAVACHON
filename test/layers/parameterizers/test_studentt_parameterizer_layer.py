import numpy as np
import tensorflow as tf

from cavachon.layers.parameterizers import StudenttParameterizerLayer


def test_shape_and_linear_reference():
    layer = StudenttParameterizerLayer(event_dims=2)
    x = tf.random.normal((3, 4), seed=7)
    out = layer(x).numpy()
    assert out.shape == (3, 6)
    assert tuple(layer.weight.shape) == (4, 6)
    assert tuple(layer.bias.shape) == (1, 6)
    np.testing.assert_allclose(
        out, x.numpy() @ layer.weight.numpy() + layer.bias.numpy(), rtol=1e-5, atol=1e-6
    )


def test_weights_registered():
    layer = StudenttParameterizerLayer(event_dims=1)
    layer(tf.ones((1, 2)))
    assert len(layer.trainable_weights) == 2


def test_get_config_contains_init_args():
    kwargs = dict(event_dims=3)
    cfg = StudenttParameterizerLayer(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


def test_from_config_round_trip():
    # regression: __init__ lacked **kwargs, so from_config (trainable/dtype) raised TypeError
    clone = StudenttParameterizerLayer.from_config(
        StudenttParameterizerLayer(**dict(event_dims=3)).get_config()
    )
    assert clone.event_dims == 3
