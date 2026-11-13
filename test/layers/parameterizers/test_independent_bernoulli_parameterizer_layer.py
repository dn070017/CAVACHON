import pytest
import numpy as np
import tensorflow as tf

from cavachon.layers.parameterizers import IndependentBernoulliParameterizerLayer


def test_build_shapes_and_output():
    layer = IndependentBernoulliParameterizerLayer(event_dims=5)
    x = tf.random.normal((3, 4), seed=0)
    out = layer(x)
    assert tuple(out.shape) == (3, 5)
    assert tuple(layer.logits_weight.shape) == (4, 5)
    assert tuple(layer.logits_bias.shape) == (1, 5)
    ref = x.numpy() @ layer.logits_weight.numpy() + layer.logits_bias.numpy()
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)


def test_batch_one_and_unbounded_logits():
    layer = IndependentBernoulliParameterizerLayer(event_dims=2)
    out = layer(tf.fill((1, 3), 1e4))
    assert tuple(out.shape) == (1, 2)
    assert np.isfinite(out.numpy()).all()


def test_get_config_contains_init_args():
    kwargs = dict(event_dims=2, name="bern")
    cfg = IndependentBernoulliParameterizerLayer(**kwargs).get_config()
    for k, v in kwargs.items():
        assert cfg[k] == v


@pytest.mark.xfail(
    strict=True,
    reason="BUG: independent_bernoulli_parameterizer_layer.py:11 __init__ takes no **kwargs, so Keras from_config (which passes "
    "trainable/dtype from get_config) raises TypeError; breaks clone_model / model save-load",
)
def test_from_config_round_trip():
    clone = IndependentBernoulliParameterizerLayer.from_config(IndependentBernoulliParameterizerLayer(**dict(event_dims=2, name="bern")).get_config())
    assert clone.event_dims == 2 and clone.name == "bern"


def test_trainable_weights():
    layer = IndependentBernoulliParameterizerLayer(event_dims=2)
    layer(tf.ones((1, 3)))
    assert len(layer.trainable_weights) == 2
