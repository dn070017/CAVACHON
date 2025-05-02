from typing import Any, Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ProgressiveScaler(tf.keras.layers.Layer):
    """ProgressiveScaler

    ProgressiveScaler used to scale the inputs during training. The
    input tensor will be scale as current_iteration/total_iteration *
    input_tensor. Do nothing in inference mode.

    Attributes
    ----------
    total_iterations: tf.Variable
        total iterations in the progressive training.

    current_iteration: tf.Variable
        current iterations in the progressive training.

    """

    def __init__(
        self,
        total_iterations: int = 5000,
        current_iteration: int = 0,
        name: str = "progressive_scaler",
        **kwargs,
    ):
        """Constructor for ProgressiveScaler

        Parameters
        ----------
        total_iterations: int, optional
            total iterations in the progressive training. Defaults to
            5000.

        current_iteration: int, optional
            current iterations in the progressive training. Default to
            0.

        name: str, optional
            Name for the tensorflow layer. Defaults to
            'progressive_scaler'.

        Raises
        ------
        ValueError
            when total_iterations is equal to or smaller than 0.

        """
        super().__init__(name=name, **kwargs)
        if total_iterations <= 0:
            raise ValueError("total_iterations must be greater than 0")

        self.total_iterations = tf.Variable(
            total_iterations * tf.ones(()), trainable=False, dtype=tf.float32
        )  # to ensure the total_iterations will be saved in the model
        self.current_iteration = tf.Variable(
            current_iteration * tf.ones(()), trainable=False, dtype=tf.float32
        )  # to ensure the total_iterations will be saved in the model

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update(
            {
                "total_iterations": self.total_iterations.numpy(),
                "current_iteration": self.current_iteration.numpy(),
            }
        )
        return config

    def call(
        self, inputs: tf.Tensor | Dict[str, tf.Tensor], training: bool = False, **kwargs
    ) -> tf.Tensor:
        """Forward pass for ProgressiveScaler

        Parameters
        ----------
        inputs: tf.Tensor
            inputs Tensor to be progressively scaled.
            Dict[str, tf.Tensor] is not supported (it merely exists for
            type checking).

        training: bool, optional
            whether to run the network in training mode. Defaults to
            False.

        Returns
        -------
        tf.Tensor
            parameters for the latent distributions.

        """
        if not isinstance(inputs, tf.Tensor):
            raise NotImplementedError

        if training:
            alpha = self.compute_alpha()
            result = alpha * inputs
            self.step()
        else:
            result = 1 * inputs

        return result

    def compute_alpha(self) -> tf.Tensor:
        """Computes the scaling factor alpha.

        Calculates alpha based on the current iteration and total
        iterations, ensuring it's capped at 1.0 and squared.

        Returns
        -------
        tf.Tensor
            the computed scaling factor alpha.
        """
        alpha = self.current_iteration / self.total_iterations
        alpha = tf.minimum(alpha, 1.0)
        alpha = alpha**2

        return alpha

    def step(self):
        """Increments the current iteration counter by one."""
        self.current_iteration.assign_add(1.0)
        self.current_iteration.assign(
            tf.minimum(self.current_iteration, self.total_iterations)
        )

    def reset(self):
        """Resets the current iteration counter to 0.0."""
        self.current_iteration.assign(0.0)
        self.current_iteration.assign(0.0)
