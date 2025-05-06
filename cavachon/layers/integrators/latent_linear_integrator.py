import warnings
from typing import Any, Dict, List

import tensorflow as tf

from cavachon.layers.integrators.progressive_scaler import ProgressiveScaler


@tf.keras.utils.register_keras_serializable()
class LatentLinearIntegrator(ProgressiveScaler):
    """LatentLinearIntegrator

    LatentLinearIntegrator used to integrate z from the component of
    interests and z_hat or z from its parent components. It expects a
    dictionary of tf.Tensor as inputs.

    """

    def __init__(
        self,
        z_key: str,
        n_latent_dims: int = 5,
        conditioned_on_keys: List[str] | None = None,
        progressive_iterations: int = 5000,
        name: str = "linear_integrator",
        **kwargs,
    ):
        """Constructor for LatentLinearIntegrator.

        Parameters
        ----------
        z_key: str
            key to access the latent representation.

        n_latent_dims: int, optional
            number of latent dimensions for the input z. Defaults to 5.

        conditioned_on_keys: List[str] | None, optional
            key to access the parent latent representation (to be
            conditioned on). Defaults to None.

        progressive_iterations: int, optional
            total iterations for progressive training. Defaults to 5000.

        name: str, optional:
            name for the tensorflow model. Defaults to
            "linear_integrator".
        """
        super().__init__(name=name, total_iterations=progressive_iterations)
        self.z_key = z_key
        self.n_latent_dims = n_latent_dims
        self.progressive_iterations = progressive_iterations
        self.r_network = tf.keras.layers.Dense(n_latent_dims)
        self.b_network = tf.keras.layers.Dense(n_latent_dims)
        if conditioned_on_keys is None:
            self.conditioned_on_keys = []
        else:
            self.conditioned_on_keys = conditioned_on_keys

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
                "z_key": self.z_key,
                "n_latent_dims": self.n_latent_dims,
                "conditioned_on_keys": self.conditioned_on_keys,
                "progressive_iterations": self.progressive_iterations,
            }
        )
        return config

    def call(
        self,
        inputs: tf.Tensor | Dict[str, tf.Tensor],
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Forward pass for LatentLinearIntegrator.

        Parameters
        ----------
        inputs: tf.Tensor | Dict[str, tf.Tensor]
            inputs Tensors for the LatentLinearIntegrator, where keys
            are 'z', 'z_conditional' (if applicable) and
            'z_hat_conditional' (if applicable). tf.Tensor is not
            supported (it merely exists for type checking).

        training: bool, optional
            whether to run the network in training mode. Defaults to
            False.

        Returns
        -------
        tf.Tensor
            z_hat (contains information of latent representation and
            the conditioned components)

        """
        if not isinstance(inputs, dict):
            raise NotImplementedError

        z = self.r_network(inputs[self.z_key])
        if training:
            alpha = self.compute_alpha()
            z = alpha * z
            self.step()

        concat_inputs = [z]
        for parent_component_key in self.conditioned_on_keys:
            if parent_component_key in inputs:
                concat_inputs.append(inputs[parent_component_key])
            else:
                warnings.warn(
                    f"{parent_component_key} is set to be integrated "
                    f"for {self.__class__.__name__} ({self.name}) "
                    "but not provided in the inputs. Do nothing.",
                    UserWarning,
                )

        z_hat = self.b_network(tf.concat(concat_inputs, axis=-1))

        return z_hat
