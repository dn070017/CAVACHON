import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.ordered_logistic_distribution import (
    OrderedLogisticDistribution,
)


class OrderedLogisticSampler(tf.keras.layers.Layer):
    """Sampler for the ordered logistic distribution."""

    def __init__(self, name: str = "ordered_logistic_sampler"):
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        score, cutpoints = OrderedLogisticDistribution.constrain(inputs)
        dist = tfp.distributions.OrderedLogistic(cutpoints=cutpoints, loc=score)
        if training:
            return tf.cast(dist.sample(), tf.float32)
        # inference: most likely state
        return tf.cast(tf.argmax(dist.categorical_probs(), axis=-1), tf.float32)
