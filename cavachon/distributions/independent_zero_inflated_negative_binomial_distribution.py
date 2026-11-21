from typing import Mapping, Union

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
        cls, params: Union[tf.Tensor, Mapping[str, tf.Tensor]], **kwargs
    ):
        """Create independent zero-inflated negative binomial
        distributions from the outputs of
        modules.parameterizers.IndependentZeroInflatedNegativeBinomial

        Parameters
        ----------
        params: Union[tf.Tensor, Mapping[str, tf.Tensor]]
            Parameters for the distribution created by parameterizers.
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

        Notes
        -----
        The zero component is a negative binomial with mean 1e-7,
        not an exact point mass at 0. Hence mean() includes a
        sigmoid(logits) * 1e-7 term and log_prob(0) is close to, but
        not exactly, log(sigmoid(logits) + (1 - sigmoid(logits)) *
        NB(0; mean, dispersion)).

        Returns
        -------
        tfp.distributions.Distribution
            Created Tensorflow Probability Zero-inflated Negative
            Binomial Distribution.

        """

        if isinstance(params, Mapping):
            logits = params.get("logits")
            mean = params.get("mean")
            dispersion = params.get("dispersion")
        else:
            params = tf.convert_to_tensor(params)
            logits, mean, dispersion = tf.split(params, 3, axis=-1)

        # the mixture weights are [sigmoid(logits), 1 - sigmoid(logits)],
        # i.e. softmax([logits, 0]). They are passed as logits so they
        # are computed in log space: with probs, 1 - sigmoid(logits)
        # rounds to 0 in float32 for logits >= ~17, which made
        # log_prob(x > 0) wrong (stuck around -78.5) and its gradient
        # w.r.t. logits NaN.
        cat_logits = tf.stack([logits, tf.zeros_like(logits)], axis=-1)

        # batch_shape: (batch, ), event_shape: (event_dims, )
        return cls(
            cat=tfp.distributions.Categorical(logits=cat_logits),
            components=(
                tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                    1e-7 * tf.ones_like(mean), dispersion
                ),
                tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                    mean, dispersion
                ),
            ),
            **kwargs,
        )
