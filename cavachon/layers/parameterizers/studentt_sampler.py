import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.studentt_distribution import StudenttDistribution


class StudenttSampler(tf.keras.layers.Layer):
    """Sampler for two-piece Student-t."""

    def __init__(self, name: str = "studentt_sampler"):
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        loc, scale, df, skewness = StudenttDistribution.constrain(inputs)
        if training:
            dist = tfp.distributions.TwoPieceStudentT(
                df=df, loc=loc, scale=scale, skewness=skewness
            )
            return dist.sample()
        return loc
