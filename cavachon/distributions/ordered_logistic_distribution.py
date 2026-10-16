import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.distribution import Distribution


class OrderedLogisticDistribution(Distribution, tfp.distributions.OrderedLogistic):
    """OrderedLogisticDistribution

    Distribution for discrete, ordered data (e.g. thresholded GISTIC2
    copy-number states -2..+2, encoded as categories 0..4).

    A single latent score per (sample, feature) is placed on a line that
    four cutpoints divide into five ordered regions. The five state
    probabilities are the slices between consecutive cutpoints, so they
    are positive, sum to one, and respect the ordering of the states by
    construction.

    """

    N_CATEGORIES = 5
    N_CUTPOINTS = N_CATEGORIES - 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    @classmethod
    def constrain(cls, params: tf.Tensor):
        """Raw parameterizer output -> (score, cutpoints).

        The only place these transforms live; the sampler calls this
        too, so the likelihood and the sampler cannot diverge.

        params layout (last dimension, size event_dims * 5):
            [0 : D]        score, one per feature
            [D : 5D]       raw cutpoints, four per feature
        """
        n_cut = cls.N_CUTPOINTS
        event_dims = tf.shape(params)[-1] // (n_cut + 1)

        score = params[..., :event_dims]
        cut_raw = params[..., event_dims:]

        # (batch, D, 4)
        cut_raw = tf.reshape(
            cut_raw, tf.concat([tf.shape(params)[:-1], [event_dims, n_cut]], axis=0)
        )

        # cutpoints must be strictly increasing: the first is free, the
        # rest are reached by positive steps.
        first = cut_raw[..., :1]
        steps = tf.math.softplus(cut_raw[..., 1:]) + 1e-3
        cutpoints = tf.concat([first, first + tf.cumsum(steps, axis=-1)], axis=-1)

        return score, cutpoints

    @classmethod
    def from_parameterizer_output(cls, params: tf.Tensor, **kwargs):
        """Create an OrderedLogistic distribution from the outputs of
        modules.parameterizers.OrderedLogistic.

        Returns
        -------
        tfp.distributions.Distribution
            batch_shape: (batch, event_dims), with 5 ordered categories.
        """
        score, cutpoints = cls.constrain(params)
        return cls(cutpoints=cutpoints, loc=score, **kwargs)
