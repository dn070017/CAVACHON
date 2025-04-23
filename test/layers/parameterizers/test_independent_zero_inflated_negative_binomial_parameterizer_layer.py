import tensorflow as tf

from cavachon.layers.parameterizers.independent_zero_inflated_negative_binomial_parameterizer_layer import (
    IndependentZeroInflatedNegativeBinomialParameterizerLayer,
)


def test_independent_zero_inflated_negative_binomial_parameterizer_layer_call():
    inputs = tf.random.normal((10, 5))
    layer = IndependentZeroInflatedNegativeBinomialParameterizerLayer(event_dims=3)
    layer.build(inputs.shape)
    outputs = layer(inputs)
    tf.debugging.assert_equal(outputs.shape, (10, 9))
