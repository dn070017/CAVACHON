import tensorflow as tf

from cavachon.layers.parameterizers.independent_bernoulli_parameterizer_layer import (
    IndependentBernoulliParameterizerLayer,
)


def test_independent_bernoulli_parameterizer_layer_call():
    inputs = tf.random.normal((10, 5))
    layer = IndependentBernoulliParameterizerLayer(event_dims=3)
    layer.build(inputs.shape)
    outputs = layer(inputs)
    tf.debugging.assert_equal(outputs.shape, (10, 3))


def test_independent_bernoulli_parameterizer_layer_get_config():
    layer = IndependentBernoulliParameterizerLayer(event_dims=3)
    config = layer.get_config()
    new_layer = IndependentBernoulliParameterizerLayer.from_config(config)
    assert new_layer.event_dims == layer.event_dims
