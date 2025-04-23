import tensorflow as tf

from cavachon.layers.parameterizers.multivariate_normal_diag_parameterizer_layer import (
    MultivariateNormalDiagParameterizerLayer,
)


def test_multivariate_normal_diag_parameterizer_layer_call():
    inputs = tf.random.normal((10, 5))
    layer = MultivariateNormalDiagParameterizerLayer(event_dims=3)
    layer.build(inputs.shape)
    outputs = layer(inputs)
    tf.debugging.assert_equal(outputs.shape, (10, 6))
