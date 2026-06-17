import tensorflow as tf

from cavachon.distributions.mixture_multivariate_normal_diag_distribution import (
    MixtureMultivariateNormalDiagDistribution,
)
from cavachon.distributions.multivariate_normal_diag_distribution import (
    MultivariateNormalDiagDistribution,
)


class GMMKLDivergence(tf.keras.losses.Loss):
    """GMMKLDivergence

    GMMKLDivergence loss adapted from Falck et al., 2021. Computes:
    logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] -
    𝚺_j𝚺_y[py_z(logpc_z)]
    """

    def __init__(
        self,
        weight_var=None,  # ← Accept a tf.Variable
        name: str = "gmm_kl_divergence", 
        **kwargs
    ):
        """Constructor for GMMKLDivergence

        Parameters
        ----------
        weight_var: float, optional
            the scaling factor for the loss. The output will be
            weight * loss. Defaults to 1.0.

        name: str, optional
            name for the tf.keras.losses.Loss (will be used when
            reporting the loss during training_step in Component and
            Model). Defaults to 'kl_divergence'.

        kwargs: Mapping[str, Any]
            additional parameters for tf.keras.losses.Loss

        """
        super().__init__(
            name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs
        )
        # Use the provided variable, or create a constant
        if weight_var is not None:
            self.weight = weight_var
        else:
            self.weight = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the GMMKLDivergence loss

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
            The computed GMMKLDivergence loss
        """
        # Based on eq (C.48) from Falck et al., 2021. Here, we use y to denote c_j
        # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y + logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpc_z)]
        # logpx_z + 𝚺_j𝚺_y[py_z(logpz_y)] + 𝚺_j𝚺_y[py_z(logpy)] - 𝚺_j[logqz_x] - 𝚺_j𝚺_y[py_z(logpy_z)]
        # can be written as:
        #   (a)   +          (b)          +          (c)        +      (d)     +          (e)
        # or
        # LogDataLikelihood - NegativeGMMKLDivergence (maximizing the ELBO)
        # or
        # NegativeLogDataLikelihood + GMMKLDivergence (minimizing the loss)

        event_dims = y_pred.shape[1] // 3
        z = y_pred[..., 0:event_dims]
        dist_z_x_params = y_pred[..., event_dims:]
        logits_prior = y_true[..., 0]

        # batch_shape: (batch, ), event_shape: (event_dims, )
        dist_z_x = MultivariateNormalDiagDistribution.from_parameterizer_output(
            dist_z_x_params
        )
        # the 1 here for dist_z_y and dist_z depends on the dimensionality of input tensor to the
        # parameterizers of MixtureMultivariateNormalDiag. In MFCVAE, this value should be one (one
        # set of priors with n_componetns for each layer)
        # batch_shape: (1, n_components), event_shape: (event_dims, )
        dist_z_y = MultivariateNormalDiagDistribution.from_parameterizer_output(
            y_true[..., 1:]
        )
        # batch_shape: (1, ), event_shape: (event_dims, )
        dist_z = MixtureMultivariateNormalDiagDistribution.from_parameterizer_output(
            y_true
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
