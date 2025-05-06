from typing import Any, Dict, List

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ModalityLinearIntegrator(tf.keras.layers.Layer):
    """ModalityLinearIntegrator

    ModalityLinearIntegrator used to integrate tensors from multiple
    modalities into a single tensor representation by robust linear
    combination.

    """

    def __init__(
        self,
        modality_keys: List[str],
        output_dims: int = 5,
        name: str = "modality_linear_integrator",
        **kwargs,
    ):
        """Constructor for ModalityLinearIntegrator.

        Parameters
        ----------
        modality_keys: List[str]
            the keys for accessing the tensors of modalities in the
            input dictionary.

        output_dims: int
            output dimension of the integrator.

        name: str, optional:
            name for the tensorflow model. Defaults to
            'modality_linear_integrator'.
        """
        super().__init__(name=name)
        self.modality_keys = modality_keys
        self.output_dims = output_dims
        self.r_networks = {
            key: tf.keras.layers.Dense(output_dims) for key in self.modality_keys
        }
        self.b_network = tf.keras.layers.Dense(output_dims)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update(
            {"modality_keys": self.modality_keys, "output_dims": self.output_dims}
        )
        return config

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        """Forward pass for ModalityLinearIntegrator.

        Parameters
        ----------
        inputs: tf.Tensor | Dict[str, tf.Tensor]


        training: bool, optional
            whether to run the network in training mode. Defaults to
            False.

        Returns
        -------
        tf.Tensor
            integrated tensor of multiple modalities using linear
            combination.

        """
        concatenated_tensors: List[tf.Tensor] = list()
        for modality_key in self.modality_keys:
            transformed_tensor = self.r_networks[modality_key](inputs[modality_key])
            concatenated_tensors.append(transformed_tensor)

        return self.b_network(tf.concat(concatenated_tensors, axis=-1))
