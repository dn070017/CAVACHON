import warnings
from typing import Any, Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ScaleConstantFactor(tf.keras.layers.Layer):
    """ScaleConstantFactor

    Modifier used to scale up the tf.Tensor with a constant factor (the
    normalized library size)

    Attributes
    ----------
    key: str
        key to access the data needed to be normalized.
    """

    def __init__(self, scaling_factor: float = 1e7, **kwargs):
        """Constructor for ScaleConstantFactor

        Parameters
        ----------
        scale_libsize: float, optional
            the scaling factor used to scale up the tf.Tensor. Defaults
            to 1e7.

        """
        super().__init__(**kwargs)
        if scaling_factor < 0.0:
            message = (
                f"invalid range for scale_libsize. Expected 0 ≤ "
                f"scale_libsize, get {scaling_factor}. Set to 1e7"
            )
            warnings.warn(message, UserWarning)
            scaling_factor = 1e7

        self.scaling_factor = scaling_factor

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"scaling_factor": self.scaling_factor})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Scale tf.Tensor with a constant scaling factor.

        Parameters
        ----------
        inputs: tf.Tensor
            input tf.Tensor.

        Returns
        -------
        tf.Tensor
            scaled tf.Tensor.

        """

        outputs = inputs * self.scaling_factor
        return outputs
