import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.distribution import Distribution


class StudenttDistribution(Distribution, tfp.distributions.TwoPieceStudentT):
    """Two-piece Student-t for continuous heavy-tailed, possibly
    asymmetric data (e.g. CNV log-ratios). skewness == 1 recovers the
    ordinary symmetric Student-t."""

    SCALE_FLOOR = 0.05
    DF_FLOOR = 1.0
    SKEW_LOG_HALFWIDTH = (
        0.7  # skewness confined to [exp(-0.7), exp(+0.7)] ~ [0.50, 2.01]
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def constrain(cls, params: tf.Tensor):
        """Raw decoder output -> valid (loc, scale, df, skewness).
        The ONLY place these transforms live. Sampler calls this too,
        so the likelihood and the sampler cannot diverge"""
        loc, scale_raw, df_raw, skew_raw = tf.split(params, 4, axis=-1)
        scale = tf.math.softplus(scale_raw) + cls.SCALE_FLOOR
        df = tf.math.softplus(df_raw) + cls.DF_FLOOR
        skewness = tf.exp(cls.SKEW_LOG_HALFWIDTH * tf.tanh(skew_raw))
        return loc, scale, df, skewness

    @classmethod
    def from_parameterizer_output(cls, params: tf.Tensor, **kwargs):
        loc, scale, df, skewness = cls.constrain(params)
        return cls(df=df, loc=loc, scale=scale, skewness=skewness, **kwargs)
