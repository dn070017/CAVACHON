from typing import Any, Dict, List, Self

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    """Decoder

    Decoder model for decoding latent representation into the
    parameters of observable data distribution.

    Attributes
    ----------
    modality_names: List[str]
        list of modality names.

    x_parameterizers: Dict[str, tf.keras.layers.Layer]
        dictionary of parameterizers for different modalities.

    n_layers: int, optional
        number of layers in the backbone network.

    n_latent_dims: int, optional
        number of latent dimensions.

    activation: str, optional
        activation function of backbone layer.

    backbone_network: tf.keras.Model
        backbone network for encoding.

    """

    def __init__(
        self,
        modality_names: List[str],
        x_parameterizers: Dict[str, tf.keras.layers.Layer],
        n_layers: int = 3,
        n_latent_dims: int = 5,
        activation: str = "swish",
        name: str = "decoder",
        **kwargs,
    ):
        """Constructor for Decoder

        Parameters
        ----------
        modality_names: List[str]
            list of modality names.

        x_parameterizers: Dict[str, tf.keras.layers.Layer]
            dictionary of parameterizers for different modalities.

        n_layers: int, optional
            number of layers in the backbone network. Defaults to 3.

        n_latent_dims: int, optional
            number of latent dimensions. Defaults to 5.

        activation: str, optional
            activation function of backbone layer. Defaults to 'swish'.

        name: str, optional
            name for the tensorflow model. Defaults to 'decoder'.
        """
        super().__init__(name=name, **kwargs)
        self.modality_names = modality_names
        self.x_parameterizers = x_parameterizers
        self.n_layers = n_layers
        self.n_latent_dims = n_latent_dims
        self.activation = activation

        self.backbone_network = TensorUtils.create_backbone_layers(
            n_layers=n_layers,
            base_n_neurons=n_latent_dims,
            reverse=False,
            activation=activation,
        )

    def call(
        self,
        inputs: Any,
        training: bool | None = None,
        mask: Any = None,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        """Forward pass of the decoder.

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
        if not isinstance(inputs, tf.Tensor):
            raise NotImplementedError(
                "inputs must be a tf.Tensor "
                f"for {self.__class__.__name__} ({self.name})"
            )

        if training is None:
            training = False

        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        for modality_name in self.modality_names:
            x_parameters = self.x_parameterizers[modality_name](outputs[modality_name])
            outputs.setdefault(
                f"{modality_name}_{Constants.MODEL_OUTPUTS_X_PARAMS}", x_parameters
            )

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the Decoder model.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update(
            {
                "modality_names": self.modality_names,
                "x_parameterizers": {
                    name: tf.keras.layers.serialize(parameterizer)
                    for name, parameterizer in self.x_parameterizers.items()
                },
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
        """Creates an Decoder model from its configuration.

        Parameters
        ----------
        config: Dict[str, Any]
            The configuration dictionary.

        Returns
        -------
        Decoder
            The Decoder model instance.
        """
        config_inputs = {k: tf.identity(v) for k, v in config.items()}
        x_parameterizers = {
            name: tf.keras.layers.deserialize(parameterizer_config)
            for name, parameterizer_config in config["x_parameterizers"].items()
        }
        config_inputs["x_parameterizers"] = x_parameterizers
        return cls(**config_inputs)

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("train_step is not implemented for Decoder.")

    def test_step(self, *args, **kwargs):
        raise NotImplementedError("test_step is not implemented for Decoder.")
