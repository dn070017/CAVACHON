import tensorflow as tf


class MultivariateNormalDiagSampler(tf.keras.layers.Layer):
    """MultivariateNormalDiagSampler

    Sampler for MultivariateNormalDiag from the outputs of Parameterizer

    """

    def __init__(
        self,
        name: str = "multivariate_normal_diag_sampler",
        stochastic: bool = True,
    ):
        """Constructor for MultivariateNormalDiagSampler

        Parameters
        ----------
        name: str, optional
            Name for the tensorflow layer. Defaults to
            'multivariate_normal_diag_sampler'.

        stochastic: bool, optional
            whether to sample in training mode. If False, the sampler
            returns the loc even in training mode. Stored as a
            non-trainable tf.Variable so the scheduler can switch it
            without retracing. Defaults to True.
        """
        super().__init__(name=name)
        self.stochastic = tf.Variable(
            1.0 if stochastic else 0.0,
            trainable=False,
            dtype=tf.float32,
            name=f"{name}_stochastic",
        )
        return

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs) -> tf.Tensor:
        """Sample for MultivariateNormalDiag from the outputs of
        Parameterizer.

        Parameters
        ----------
        inputs: tf.Tensor
            inputs Tensor.

        training: bool, optional
            if the layer is called in training mode. Defaults to False.

        Returns
        -------
        tf.Tensor
            sampling results.
        """
        loc, scale_diag = tf.split(inputs, 2, axis=-1)
        if training and self.trainable:
            epsilon = tf.random.normal(shape=tf.shape(loc))
            return self.stochastic * epsilon * scale_diag + loc
        else:
            return loc
