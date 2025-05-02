import tensorflow as tf

from cavachon.layers.parameterizers.mixture_multivariate_normal_diag_parameterizer_layer import (
    MixtureMultivariateNormalDiagParameterizerLayer,
)


def test_mixture_multivariate_normal_diag_parameterizer_layer_call():
    inputs = tf.random.normal((10, 5))
    layer = MixtureMultivariateNormalDiagParameterizerLayer(
        event_dims=3, n_components=2
    )
    layer.build(inputs.shape)
    outputs = layer(inputs)
    tf.debugging.assert_equal(outputs.shape, (10, 2, 7))


def test_mixture_multivariate_normal_diag_parameterizer_layer_get_config():
    layer = MixtureMultivariateNormalDiagParameterizerLayer(
        event_dims=3, n_components=2
    )
    config = layer.get_config()
    new_layer = MixtureMultivariateNormalDiagParameterizerLayer.from_config(config)
    assert new_layer.event_dims == layer.event_dims
    assert new_layer.n_components == layer.n_components
