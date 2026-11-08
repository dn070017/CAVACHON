import tensorflow as tf

from cavachon.layers.progressive_scaler import ProgressiveScaler


class StandardKLDivergence(tf.keras.losses.Loss):
    """StandardKLDivergence

    KL divergence between encoder's N(μ, σ²) and standard normal N(0, 1).
    KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum[1 + log(σ²) - μ² - σ²]

    The encoder (MultivariateNormalDiagParameterizerLayer) outputs the
    standard deviation σ = softplus(·) + 1e-7, not log(σ²).

    """

    def __init__(
        self, 
        weight: float = 3.0,
        name: str = "standard_kl_divergence", 
        **kwargs
    ):
        """Standard KL with progressive scaler: effective weight = scale × α²."""
        super().__init__(
            name=name,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            **kwargs,
        )
        self.weight = ProgressiveScaler(
            total_iterations=1, scale=float(weight),
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
        # y_pred layout: [sampled_z, mean, std]
        event_dims = y_pred.shape[1] // 3

        # Step 2: Extract mean (μ) and std (σ) from encoder output
        # We skip the first part (sampled z) and only use mean and std
        mean = y_pred[..., event_dims : 2 * event_dims]  # μ
        std = y_pred[..., 2 * event_dims : 3 * event_dims]  # σ

        # Step 3: calculate variance and log-variance
        # std >= 1e-7 (softplus + 1e-7 in the parameterizer), so log is safe
        var = tf.square(std)  # σ²
        log_var = 2.0 * tf.math.log(std)  # log(σ²)

        # Step 4: Apply the closed-form KL formula
        # KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum[1 + log(σ²) - μ² - σ²]
        # Closed form is >= 0 since σ² >= 1 + log(σ²), so no clamping
        # is needed (clamping would also zero the gradients).
        kl_divergence = -0.5 * tf.reduce_sum(
            1.0 + log_var - tf.square(mean) - var,
            axis=-1,  # Sum over latent dimensions
        )
        # Result shape: (batch,) - one KL value per sample

        # Step 5: Scale by weight and return
        # The reduction=SUM_OVER_BATCH_SIZE averages this automatically
        return self.weight(tf.ones(())) * kl_divergence
