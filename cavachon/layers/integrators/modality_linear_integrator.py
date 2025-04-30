from typing import Dict, List

import tensorflow as tf


class ModalityLinearIntegrator(tf.keras.layers.Layer):
    """ModalityLinearIntegrator

    ModalityLinearIntegrator used to integrate tensors from multiple
    modalities into a single tensor representation by linear
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
        self.r_network = tf.keras.layers.Dense(output_dims)

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
            concatenated_tensors.append(inputs[modality_key])

        return self.r_network(tf.concat(concatenated_tensors, axis=-1))
