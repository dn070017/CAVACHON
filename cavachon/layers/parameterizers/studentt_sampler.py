import tensorflow as tf
import tensorflow_probability as tfp


class StudenttSampler(tf.keras.layers.Layer):
    """Sampler for Student-T distribution."""

    def __init__(self, name: str = "studentt_sampler"):
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """
        Samples from the Student-T distribution using the parameters.
        """
        # We split the 3 parameters (loc, scale, df)
        loc, raw_scale, raw_df = tf.split(inputs, 3, axis=-1)

        # Apply the same constraints as in the Distribution class
        scale = tf.math.softplus(raw_scale) + 1e-7
        df = tf.math.softplus(raw_df) + 2.0

        if training:
            # During training, we use TFP's sample method (Reparameterization trick)
            dist = tfp.distributions.StudentT(df=df, loc=loc, scale=scale)
            return dist.sample()
        else:
            # During inference/prediction, we usually just return the mean (loc)
            return loc
