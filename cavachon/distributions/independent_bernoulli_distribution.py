from typing import Mapping, Self

import tensorflow as tf
import tensorflow_probability as tfp

from cavachon.distributions.distribution import Distribution


class IndependentBernoulliDistribution(Distribution, tfp.distributions.Bernoulli):
    """IndependentBernoulliDistribution

    Distribution for independent Bernoulli.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    @classmethod
    def from_parameterizer_output(
        cls, params: tf.Tensor | Mapping[str, tf.Tensor], **kwargs
    ) -> Self:
        """Create independent Bernoulli distributions from the outputs
        of modules.parameterizers.IndependentBernoulli.

        Parameters
        ----------
        params: tf.Tensor | Mapping[str, tf.Tensor]
            parameters for the distribution created by parameterizers.
            Alternatively, a mapping of tf.Tensor with parameter name
            as keys can be provided. If provided with a tf.Tensor, it
            will be used as the logits to create Bernoulli distribution.
            If provided with a Mapping, 'logits' should be in the keys
            of the Mapping. Note that:
            1.  The batch_shape should be the same as the provided
                tf.Tensor.
            2.  The event_shape should be [].

        Returns
        -------
        IndependentBernoulliDistribution
            created Tensorflow Probability Bernoulli Distribution.

        """
        if isinstance(params, tf.Tensor):
            logits = params
        elif isinstance(params, Mapping):
            logits = params["logits"]

        # batch_shape: (batch, ), event_shape: []
        return cls(logits=logits, **kwargs)
