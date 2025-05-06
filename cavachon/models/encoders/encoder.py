from typing import Any, Dict, List, Self

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.integrators.modality_linear_integrator import (
    ModalityLinearIntegrator,
)
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)
from cavachon.layers.parameterizers.multivariate_normal_diag_parameterizer_layer import (
    MultivariateNormalDiagParameterizerLayer,
)
from cavachon.layers.parameterizers.multivariate_normal_diag_sampler import (
    MultivariateNormalDiagSampler,
)
from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    """Encoder

    Encoder model for encoding input data into latent representations.

    Attributes
    ----------
    modality_names: List[str]
        list of modality names.

    modifiers: Dict[str, DistributionPresetModifier]
        dictionary of modifiers for different modalities.

    n_reduced_dims: int, optional
        number of reduced dimensions

    n_layers: int, optional
        number of layers in the backbone network.

    n_latent_dims: int, optional
        number of latent dimensions.

    activation: str, optional
        activation function of backbone layer.

    modifiers_backbone_adaptor: ModalityLinearIntegrator
        integrator for modality tensors.

    backbone_network: tf.keras.Model
        backbone network for encoding.

    z_parameterizer: MultivariateNormalDiagParameterizerLayer
        parameterizer for latent variables.

    z_sampler: MultivariateNormalDiagSampler
        sampler for latent variables.

    """

    def __init__(
        self,
        modality_names: List[str],
        modifiers: Dict[str, DistributionPresetModifier],
        n_reduced_dims: int = 512,
        n_layers: int = 3,
        n_latent_dims: int = 5,
        activation: str = "swish",
        name: str = "encoder",
        **kwargs,
    ):
        """Constructor for Encoder

        Parameters
        ----------
        modality_names: List[str]
            list of modality names.

        modifiers: Dict[str, DistributionPresetModifier]
            dictionary of modifiers for different modalities.

        n_reduced_dims: int, optional
            number of reduced dimensions. Defaults to 512.

        n_layers: int, optional
            number of layers in the backbone network. Defaults to 3.

        n_latent_dims: int, optional
            number of latent dimensions. Defaults to 5.

        activation: str, optional
            activation function of backbone layer. Defaults to 'swish'.

        name: str, optional
            name for the tensorflow model. Defaults to 'encoder'.
        """
        super().__init__(name=name, **kwargs)
        self.modality_names = modality_names
        self.modifiers = modifiers
        self.n_reduced_dims = n_reduced_dims
        self.n_layers = n_layers
        self.n_latent_dims = n_latent_dims
        self.activation = activation

        self.modifiers_backbone_adaptor = ModalityLinearIntegrator(
            modality_keys=[
                f"{modality}_{Constants.TENSOR_NAME_X_MODEL}"
                for modality in modality_names
            ],
            output_dims=n_reduced_dims,
        )
        self.backbone_network = TensorUtils.create_backbone_layers(
            n_layers=n_layers,
            base_n_neurons=n_latent_dims,
            reverse=True,
            activation=activation,
        )
        self.z_parameterizer = MultivariateNormalDiagParameterizerLayer(
            event_dims=n_latent_dims
        )
        self.z_sampler = MultivariateNormalDiagSampler()

    def preprocess(
        self, inputs: Dict[str, tf.Tensor], **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Preprocess the tensor.

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            input tensors.

        Returns
        -------
        Dict[str, tf.Tensor]
            preprocessed tensor.

        """
        for _, modifier in self.modifiers.items():
            inputs = modifier(inputs)

        return inputs

    def transform(
        self, inputs: Dict[str, tf.Tensor], training: bool = False, **kwargs
    ) -> tf.Tensor:
        """Transform the processed tensor into the encoded tensor.

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            input tensors.

        training: bool, optional
            whether the call is during training. Defaults to False.

        Returns
        -------
        tf.Tensor
            encoded tensor.

        """
        modality_integrated_tensors = self.modifiers_backbone_adaptor(
            inputs, training=training
        )
        encoded_tensor = self.backbone_network(
            modality_integrated_tensors, training=training
        )
        return encoded_tensor

    def parameterize(
        self, inputs: tf.Tensor, training: bool = False, **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Transform the encoded tensor into latent variables by
        parameterization.

        Parameters
        ----------
        inputs: tf.Tensor
            encoded tensor.

        training: bool, optional
            whether the call is during training. Defaults to False.

        Returns
        -------
        Dict[str, tf.Tensor]
            output tensors.

        """
        z_parameters = self.z_parameterizer(inputs, training=training)
        z = self.z_sampler(z_parameters, training=training)
        return {
            Constants.MODEL_OUTPUTS_Z: z,
            Constants.MODEL_OUTPUTS_Z_PARAMS: z_parameters,
        }

    def call(
        self,
        inputs: Any,
        training: bool | None = None,
        mask: Any = None,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        """Forward pass of the encoder.

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            input tensors.

        training: bool | None, optional
            whether the call is during training. Defaults to False.

        Returns
        -------
        Dict[str, tf.Tensor]
            output tensors.

        """
        if not isinstance(inputs, dict):
            raise NotImplementedError(
                "inputs must be a dictionary with string keys and tf.Tensor values ",
                f"for {self.__class__.__name__} ({self.name})",
            )

        if training is None:
            training = False

        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        outputs = self.preprocess(outputs)
        encoded_tensor = self.transform(outputs, training=training)
        outputs_z = self.parameterize(encoded_tensor, training=training)

        outputs[Constants.MODEL_OUTPUTS_Z] = outputs_z[Constants.MODEL_OUTPUTS_Z]
        outputs[Constants.MODEL_OUTPUTS_Z_PARAMS] = outputs_z[
            Constants.MODEL_OUTPUTS_Z_PARAMS
        ]

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the Encoder model.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "modality_names": self.modality_names,
                "modifiers": {
                    name: tf.keras.layers.serialize(modifier)
                    for name, modifier in self.modifiers.items()
                },
                "n_reduced_dims": self.n_reduced_dims,
                "n_layers": self.n_layers,
                "n_latent_dims": self.n_latent_dims,
                "activation": self.activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects: Any | None = None
    ) -> Self:
        """Creates an Encoder model from its configuration.

        Parameters
        ----------
        config: Dict[str, Any]
            The configuration dictionary.

        Returns
        -------
        Encoder
            The Encoder model instance.
        """
        config_inputs = {k: tf.identity(v) for k, v in config.items()}
        modifiers = {
            name: tf.keras.layers.deserialize(modifier_config)
            for name, modifier_config in config["modifiers"].items()
        }
        config_inputs["modifiers"] = modifiers
        return cls(**config_inputs)

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("train_step is not implemented for Encoder.")

    def test_step(self, *args, **kwargs):
        raise NotImplementedError("test_step is not implemented for Encoder.")
