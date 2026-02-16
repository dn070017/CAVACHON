import tensorflow as tf


class VanillaKLDivergence(tf.keras.losses.Loss):
    """VanillaKLDivergence

    KL divergence between encoder's N(μ, σ²) and standard normal N(0, 1).
    KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum[1 + log(σ²) - μ² - σ²]

    - In order to avoid exploding graients, let log(σ²) be the direct output 's'
    - Then σ² = exp(s)

    """

    def __init__(
        self, weight: float = 3.0, name: str = "vanilla_kl_divergence", **kwargs
    ):
        """Constructor for VanillaKLDivergence

        Parameters
        ----------
        weight: float, optional
            Scaling factor for the loss. Defaults to 1.0.

        name: str, optional
            Name for the loss (shows up in training logs). Defaults to 'vanilla_kl_divergence'.
        """
        self.weight = weight
        super().__init__(
            name=name,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,  # Average over batch
            **kwargs,
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the vanilla KL divergence loss

        Parameters
        ----------
        y_true: tf.Tensor
            Not used (prior is fixed N(0,1)), but kept for TensorFlow API compatibility.

        y_pred: tf.Tensor
            Shape: (batch, event_dims * 3)
            Contains: [sampled_z, mean, std] concatenated
            - y_pred[..., 0:event_dims] = sampled z (we ignore this)
            - y_pred[..., event_dims:2*event_dims] = mean (μ) from encoder
            - y_pred[..., 2*event_dims:3*event_dims] = std (σ) from encoder

        Returns
        -------
        tf.Tensor:
            The KL divergence value (positive scalar, averaged over batch)
        """

        # Step 1: Figure out dimensions
        # y_pred has 3 parts concatenated, so divide by 3
        # y_pred layout: [sampled_z, mean, log_var]
        event_dims = y_pred.shape[1] // 3

        # Step 2: Extract mean (μ) and std (σ) from encoder output
        # We skip the first part (sampled z) and only use mean and std
        mean = y_pred[..., event_dims : 2 * event_dims]  # μ
        log_var = y_pred[..., 2 * event_dims : 3 * event_dims]  # log(σ²)

        # Step 3: calculate variance
        # use exp(), so that var is guaranteed to be positive
        var = tf.exp(log_var)
        # var = tf.square(std)  # σ² = std²
        # log_var = tf.math.log(var + 1e-7)  # log(σ²), add small number to avoid log(0)

        # Step 4: Apply the closed-form KL formula
        # KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum[1 + log(σ²) - μ² - σ²]
        kl_divergence = -0.5 * tf.reduce_sum(
            1.0 + log_var - tf.square(mean) - var,
            axis=-1,  # Sum over latent dimensions
        )
        # Result shape: (batch,) - one KL value per sample

        # Step 5: Ensure KL is non-negative
        # (Sometimes numerical errors make it slightly negative)
        kl_divergence = tf.maximum(kl_divergence, 0.0)
        # kl_divergence = tf.where(
        #    kl_divergence < 0, tf.zeros_like(kl_divergence), kl_divergence
        # )

        # Step 6: Scale by weight and return
        # The reduction=SUM_OVER_BATCH_SIZE averages this automatically
        return self.weight * kl_divergence
