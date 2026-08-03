from typing import Optional, Union

import tensorflow as tf

from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.layers.progressive_scaler import ProgressiveScaler


class GMMDensityLoss(tf.keras.losses.Loss):
    """GMMDensityLoss

    Batch-mean negative log-likelihood of deterministic z_hat values
    under a learnable diagonal Gaussian Mixture Model (GMM) prior.

    Loss = -mean(log p(z_hat | GMM_prior))

    where p(z_hat) = sum_k[ pi_k * N(z_hat | mu_k, diag(sigma_k^2)) ].

    When a ``weight`` is provided, the loss is multiplied by it before
    being returned.  The reported metric should divide by the weight
    to recover the raw (unweighted) negative log-density.
    """

    def __init__(
        self,
        weight: Optional[Union[tf.Variable, ProgressiveScaler]] = None,
        name: str = "gmm_density_loss",
        **kwargs,
    ):
        """Initialize the GMM density loss.

        Parameters
        ----------
        weight: tf.Variable or ProgressiveScaler, optional
            External weight multiplier. The scheduler toggles this
            between 0.0 (disabled) and 1.0 (enabled) to gate the
            loss during different training phases.  Defaults to None
            (no weighting).

        name: str, optional
            Name for the loss function. Defaults to
            'gmm_density_loss'.
        """
        super().__init__(
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            **kwargs,
        )
        self.weight = weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the batch-mean negative log-likelihood of z_hat under
        a learnable diagonal GMM prior.

        Parameters
        ----------
        y_true: tf.Tensor
            The outputs of
            layers.parameterizers.MixtureMultivariateNormalDiag
            with shape (batch, n_components, event_dims * 2 + 1),
            where:
            1.  y_true[..., 0] are the logits,
            2.  y_true[..., 1:event_dims+1] are the loc (mean) for
                each component,
            3.  y_true[..., event_dims+1:] are the scale_diag (std)
                for each component.

        y_pred: tf.Tensor
            Deterministic z_hat values with shape
            (batch, event_dims).

        Returns
        -------
        tf.Tensor:
            A scalar tensor representing the batch-mean negative
            log-likelihood, multiplied by ``self.weight`` (if set).
        """
        # Build the GMM prior distribution from parameterizer output.
        dist = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
            y_true
        )
        # Compute per-sample log-probability.
        log_prob = dist.log_prob(y_pred)
        # Batch-mean negative log-likelihood.
        loss = -tf.reduce_mean(log_prob)
        if self.weight is not None:
            loss = loss * self.weight
        return loss
