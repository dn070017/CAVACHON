from typing import Mapping, Self

import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.distribution import Distribution


class IndependentZeroInflatedNegativeBinomialDistribution(
    Distribution, tfp.distributions.Mixture
):
    """IndependentZeroInflatedNegativeBinomialDistribution

    Distribution for independent zero-inflated negative binomial.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    @classmethod
    def from_parameterizer_output(
        cls, params: tf.Tensor | Mapping[str, tf.Tensor], **kwargs
    ) -> Self:
        """Create independent zero-inflated negative binomial
        distributions from the outputs of
        modules.parameterizers.IndependentZeroInflatedNegativeBinomial

        Parameters
        ----------
        params: tf.Tensor | Mapping[str, tf.Tensor]
            parameters for the distribution created by parameterizers.
            Alternatively, a mapping of tf.Tensor with parameter name
            as keys can be provided. If provided with a tf.Tensor. The
            last dimension needs to be a multiple of 3, and:
            1.  params[..., 0:p] will be used as the logits.
            2.  params[..., p:2*p] will be used as the mean.
            3.  params[..., 2*p:3*p] will be used as the dispersion.
            If provided with a Mapping, 'logits', 'mean', 'dispersion'
            should be in the keys of the Mapping. Note that in both
            cases:
            1.  The batch_shape should be the same as [..., p].
            2.  The event_shape should be [].

        Returns
        -------
        IndependentZeroInflatedNegativeBinomialDistribution
            created Tensorflow Probability Zero-inflated Negative
            Binomial Distribution.

        """

        if isinstance(params, tf.Tensor):
            logits, mean, dispersion = tf.split(params, 3, axis=-1)
        elif isinstance(params, Mapping):
            logits = params["logits"]
            mean = params["mean"]
            dispersion = params["dispersion"]

        probs = tf.math.sigmoid(logits)
        probs = tf.stack([probs, 1 - probs], axis=-1)

        # batch_shape: (batch, ), event_shape: []
        return cls(
            cat=tfp.distributions.Categorical(probs=probs),
            components=(
                tfp.distributions.Deterministic(tf.zeros_like(mean)),
                tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                    mean, dispersion
                ),
            ),
            **kwargs,
        )
