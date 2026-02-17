import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.distribution import Distribution


class StudentTDistribution(Distribution, tfp.distributions.StudentT):
    """StudentT distribution for continuous data with heavy tails (e.g. CNV)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_parameterizer_output(cls, params: tf.Tensor, **kwargs):
        """
        Creates distribution from a single tensor.
        The last dimension is split into 3: loc, scale, and df.
        """
        # Split into 3 equal parts
        loc, scale_raw, df_raw = tf.split(params, 3, axis=-1)

        # Scale (sigma) must be positive
        scale = tf.math.softplus(scale_raw) + 1e-7

        # Degrees of Freedom (nu) must be > 0.
        # Adding 2.0 ensures the variance is mathematically defined (> 2).
        df = tf.math.softplus(df_raw) + 2.0

        return cls(df=df, loc=loc, scale=scale, **kwargs)
