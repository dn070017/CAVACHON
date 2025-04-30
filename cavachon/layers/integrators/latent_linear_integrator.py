import warnings
from typing import Dict

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.integrators.progressive_scaler import ProgressiveScaler


class LatentLinearIntegrator(ProgressiveScaler):
    """LatentLinearIntegrator

    LatentLinearIntegrator used to integrate z from the component of
    interests and z_hat or z from its parent components. It expects a
    dictionary of tf.Tensor as inputs. The key of the inputs are 'z',
    'z_conditional' (if applicable) and 'z_hat_conditional'
    (if applicable).

    """

    def __init__(
        self,
        n_latent_dims: int = 5,
        is_conditioned_on_z: bool = False,
        is_conditioned_on_z_hat: bool = False,
        progressive_iterations: int = 5000,
        name: str = "linear_integrator",
        **kwargs,
    ):
        """Constructor for LatentLinearIntegrator.

        Parameters
        ----------
        n_latent_dims: int, optional
            number of latent dimensions for the input z. Defaults to 5.

        is_conditioned_on_z: bool, optional
            use latent representation from the conditioned components.
            Defaults to False.

        is_conditioned_on_z_hat: bool, optional
            use transformed latent representation (contains information
            of all ancestor of conditioned components) from the
            conditioned components. Defaults to False.

        progressive_iterations: int, optional
            total iterations for progressive training. Defaults to 5000.

        name: str, optional:
            name for the tensorflow model. Defaults to
            "linear_integrator".
        """
        super().__init__(name=name, total_iterations=progressive_iterations)
        self.is_conditioned_on_z = is_conditioned_on_z
        self.is_conditioned_on_z_hat = is_conditioned_on_z_hat
        self.r_network = tf.keras.layers.Dense(n_latent_dims)
        self.b_network = tf.keras.layers.Dense(n_latent_dims)

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

        z = self.r_network(inputs[Constants.MODEL_OUTPUTS_Z])
        if training:
            alpha = self.compute_alpha()
            z = alpha * z
            self.step()

        concat_inputs = [z]

        if self.is_conditioned_on_z or self.is_conditioned_on_z_hat:
            if self.is_conditioned_on_z:
                if Constants.MODULE_INPUTS_CONDITIONED_Z in inputs:
                    concat_inputs.append(inputs[Constants.MODULE_INPUTS_CONDITIONED_Z])
                else:
                    warnings.warn(
                        f"is_conditioned_on_z is set for {self.__class__.__name__} ({self.name}) "
                        "but not provided in the inputs. Do nothing.",
                        UserWarning,
                    )
            if self.is_conditioned_on_z_hat:
                if Constants.MODULE_INPUTS_CONDITIONED_Z_HAT in inputs:
                    concat_inputs.append(
                        inputs[Constants.MODULE_INPUTS_CONDITIONED_Z_HAT]
                    )
                else:
                    warnings.warn(
                        f"is_conditioned_on_z_hat is set for {self.__class__.__name__} ({self.name}) "
                        "but not provided in the inputs. Do nothing.",
                        UserWarning,
                    )

        z_hat = self.b_network(tf.concat(concat_inputs, axis=-1))

        return z_hat
