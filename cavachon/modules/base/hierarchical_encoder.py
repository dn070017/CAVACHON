from typing import Optional

import numpy as np
import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.parameterizers.multivariate_normal_diag_parameterizer_layer import (
    MultivariateNormalDiagParameterizerLayer,
)
from cavachon.layers.progressive_scaler import ProgressiveScaler


class HierarchicalEncoder(tf.keras.Model):
    """HierarchicalEncoder

    HierarchicalEncoder used to encode z_hat hierarchically through
    the dependency between components. It expects multiple tf.Tensor
    as inputs. The key of the inputs are 'z', 'z_conditional' (if
    applicable) and 'z_hat_conditional' (if applicable). The outputs
    are the parameters (loc and scale_diag) of the independent
    Gaussian distribution of z_hat. The loc is a linear function of
    the inputs. This base module is implemented using Tensorflow
    sequential API.

    """

    def __init__(
        self,
        n_latent_dims: int = 5,
        is_conditioned_on_z: bool = False,
        is_conditioned_on_z_hat: bool = False,
        progressive_iterations: int = 5000,
        use_bias: bool = False,
        name: str = "hierarchical_encoder",
        **kwargs,
    ):
        """Constructor for HierarchicalEncoder. Should not be called
        directly most of the time. Please use make() to create the
        model.

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

        use_bias: bool, optional
            whether to use bias in the Dense layer of r_network. The
            parameterizer in b_network always uses bias. Defaults to
            False.

        name: str, optional:
            Name for the tensorflow model. Defaults to
            'hierarchical_encoder'.
        """
        super().__init__(name=name)
        self.is_conditioned_on_z = is_conditioned_on_z
        self.is_conditioned_on_z_hat = is_conditioned_on_z_hat
        self.progressive_scaler = ProgressiveScaler(progressive_iterations)
        self.r_network = tf.keras.Sequential(
            [tf.keras.layers.Dense(n_latent_dims, use_bias=use_bias, name=f"{name}_r_dense")],
            name=Constants.MODULE_R_NETWORK
        )
        # loc_weight is the first weight of the parameterizer, so
        # b_network.get_weights()[0] remains the linear map of the loc.
        self.z_hat_parameterizer = MultivariateNormalDiagParameterizerLayer(
            n_latent_dims, name=f"{name}_b_parameterizer"
        )
        self.b_network = tf.keras.Sequential(
            [self.z_hat_parameterizer],
            name=Constants.MODULE_B_NETWORK
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Forward pass for HierarchicalEncoder.

        Parameters
        ----------
        inputs: Mapping[str, tf.Tensor]
            inputs Tensors for the HierarchicalEncoder, where keys are
            'z', 'z_conditional' (if applicable) and 'z_hat_conditional'
            (if applicable). (if applicable). (by defaults, expect the
            outputs by HierarchicalEncoder)

        training: bool, optional
            whether to run the network in training mode. Defaults to
            False.

        mask: tf.Tensor, optional
            a mask or list of masks. Defaults to None.

        Returns
        -------
        tf.Tensor
            parameters of z_hat (contains information of latent
            representation and the conditioned components), with shape
            (batch, n_latent_dims * 2), where
            1.  results[..., 0:n_latent_dims] are the loc (mean)
            2.  results[..., n_latent_dims:] are the scale_diag (std)

        """
        z_hat = self.r_network(inputs.get(Constants.MODEL_OUTPUTS_Z))
        z_hat = self.progressive_scaler(z_hat)
        concat_inputs = []
        if self.is_conditioned_on_z or self.is_conditioned_on_z_hat:
            if self.is_conditioned_on_z:
                concat_inputs.append(
                    inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z, None)
                )
            if self.is_conditioned_on_z_hat:
                concat_inputs.append(
                    inputs.get(Constants.MODULE_INPUTS_CONDITIONED_Z_HAT, None)
                )

        concat_inputs.append(z_hat)
        z_hat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(concat_inputs)
        z_hat_parameters = self.b_network(z_hat)

        return z_hat_parameters

    def initialize_scale_diag(self, scale_diag: np.ndarray) -> None:
        """Set the scale_diag of z_hat to a constant per dimension by
        zeroing the scale weights and assigning the bias to the
        inverse softplus of scale_diag.

        Parameters
        ----------
        scale_diag: np.ndarray
            target std for each latent dimension, with shape
            (n_latent_dims, ).
        """
        scale_diag = np.maximum(np.asarray(scale_diag, dtype=np.float32), 1e-6)
        # inverse softplus: log(exp(y) - 1) = y + log(1 - exp(-y))
        bias = scale_diag + np.log(-np.expm1(-scale_diag))
        parameterizer = self.z_hat_parameterizer
        parameterizer.scale_diag_weight.assign(
            tf.zeros_like(parameterizer.scale_diag_weight)
        )
        parameterizer.scale_diag_bias.assign(bias.reshape(1, -1))

    def train_step(self, data):
        raise NotImplementedError()
