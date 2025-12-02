import tensorflow as tf

from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)


class KLDivergence(tf.keras.losses.Loss):
    """KLDivergence

    KLDivergence loss adapted from Falck et al., 2021. Computes:
    logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] -
    𝚺_j𝚺_y[py_z(logpc_z)]
    """

    def __init__(
        self,
        event_dims: int,
        n_cluster: int,
        weight: float = 1.0,
        name: str = "kl_divergence",
        **kwargs,
    ):
        """Constructor for KLDivergence

        Parameters
        ----------
        weight: float, optional
            the scaling factor for the loss. The output will be
            weight * loss. Defaults to 1.0.

        name: str, optional
            name for the tf.keras.losses.Loss (will be used when
            reporting the loss during training_step in Component and
            Model). Defaults to 'kl_divergence'.

        kwargs: Mapping[str, Any]
            additional parameters for tf.keras.losses.Loss

        """
        self.weight = weight
        self.event_dims = event_dims
        self.n_cluster = n_cluster
        super().__init__(
            name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the KLDivergence loss

        Parameters
        ----------
        y_true: tf.Tensor
            The outputs of
            layers.parameterizers.MixtureMultivariateNormalDiag with a
            inputs of tf.ones((1, 1)), which outputs a tf.Tensor with a
            shape of (1, n_components, event_dims * 2 + 1), where:
            1.  y_true[..., 0] is the logits for mixture distribution,
            2.  y_true[..., 1:event_dims+1] is the locs for each
                distribution.
            3.  y_true[..., event_dims+1:] is the scale_diag for each
                distribution.
            Note that this special requirement is designed to follow
            the API tf.keras.losses.Loss provides. Can be ignored if
            the developers wish to use custom eager training.

        y_pred: tf.Tensor
            The outputs of layers.parameterizers.MultivariateNormalDiag,
            which outputs a tf.Tensor with a shape of
            (batch, event_dims * 2), where
            1. y_pred[..., 0:event_dim] is the loc.
            2. y_pred[..., event_dim:2*event_dims] is the scale_diag.
            Note that this special requirement is designed to follow
            the API tf.keras.losses.Loss provides. Can be ignored if
            the developers wish to use custom eager training.

        Returns
        -------
        tf.Tensor:
            The computed KLDivergence loss
        """
        # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
        # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)]
        # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)]
        # can be written as:
        #   (a)   +          (b)          +          (c)        +      (d)     +          (e)
        # or
        # LogDataLikelihood - NegativeKLDivergence (maximizing the ELBO)
        # or
        # NegativeLogDataLikelihood + KLDivergence (minimizing the loss)

        # y_true.shape = (n_cluster, 2 * event_dims + 1)
        # y_pred.shape = (batch, event_dims * 3)
        # y_true_transform = (batch, n_cluster * (2 * event_dims + 1))
        # y_pred_new = tf.concat([y_pred, y_true], axis=1)
        # y_pred_new = (batch, event_dims * 3 + n_cluster * (2 * event_dims + 1))

        # logits_prior = y_true[..., 0]
        event_dims = self.event_dims
        # we split y_pred into 2, first part posterior, second part prior
        y_pred_posterior, y_pred_prior = tf.split(
            y_pred, [event_dims * 3, y_pred.shape[1] - event_dims * 3], axis=1
        )
        # posterior = first 3 * event_dims entries
        z = y_pred_posterior[..., 0:event_dims]  # z(latent space)
        dist_z_x_params = y_pred_posterior[
            ..., event_dims:
        ]  # mean and std of posterior
        # y_pred_prior needs to reshape back into its original form (currently it is flattened)
        # because for each cluster we need event dims (mean and std) + 1 logit
        y_pred_prior = tf.reshape(
            y_pred_prior, (1, self.n_clusters, 2 * event_dims + 1)
        )  # corrected to tf.reshape
        logits_prior = y_pred_prior[
            ..., 0
        ]  # extracts all mixture logits (one per cluster)

        # batch_shape: (batch, ), event_shape: (event_dims, )
        dist_z_x = MultivariateNormalDiagDistribution.from_parameterizer_output(
            dist_z_x_params
        )
        # the 1 here for dist_z_y and dist_z depends on the dimensionality of input tensor to the
        # parameterizers of MixtureMultivariateNormalDiag. In MFCVAE, this value should be one (one
        # set of priors with n_componetns for each layer)
        # batch_shape: (1, n_components), event_shape: (event_dims, )
        dist_z_y = MultivariateNormalDiagDistribution.from_parameterizer_output(
            y_pred_prior[..., 1:]
        )
        # batch_shape: (1, ), event_shape: (event_dims, )
        dist_z = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
            y_pred_prior
        )

        # change the shape of z from (batch, event_dims) to (batch, 1, event_dims) to make the
        # operation broadcastable with batch shape (1, n_components) of dist_z_y
        # shape: (batch, n_components)
        logpz_y = dist_z_y.log_prob(tf.expand_dims(z, -2))
        # shape: (1, n_components)
        logpy = tf.math.log(tf.math.softmax(logits_prior) + 1e-7)
        # shape: (batch, 1)
        logpz = tf.expand_dims(dist_z.log_prob(z), -1)

        # shape: (batch, n_components)
        py_z = tf.math.softmax(logpz_y + logpy - logpz)
        logpy_z = tf.math.log(py_z + 1e-7)
        # logpy_z = logpz_y + logpy - logpz
        # py_z = tf.exp(logpy_z)

        # term (b): 𝚺_j𝚺_y[py_z(logpz_y)]
        py_z_logpz_y = tf.reduce_sum(py_z * logpz_y, axis=-1)

        # term (c): 𝚺_j𝚺_y[py_z(logpy)]
        py_z_logpy = tf.reduce_sum(py_z * logpy, axis=-1)

        # term (d): 𝚺_j[logqz_x]
        logqz_x = dist_z_x.log_prob(z)

        # term (e): 𝚺_j𝚺_y[py_z(logpy_z)]
        py_z_logpy_z = tf.reduce_sum(py_z * logpy_z, axis=-1)

        kl_divergence = -py_z_logpz_y - py_z_logpy + py_z_logpy_z + logqz_x
        kl_divergence = tf.where(
            kl_divergence < 0, tf.zeros_like(kl_divergence), kl_divergence
        )

        return self.weight * kl_divergence
