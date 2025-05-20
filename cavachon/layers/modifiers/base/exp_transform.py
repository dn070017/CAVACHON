import math
from typing import Any, Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ExpTransform(tf.keras.layers.Layer):
    """ExpTransform

    Modifier used to exponential transform the tf.Tensor.

    Attributes
    ----------
    base: float
        the base used for exponential transform.

    """

    def __init__(
        self,
        base: float = math.e,
        **kwargs,
    ):
        """Constructor for ExpTransform

        Parameters
        ----------
        base: float, optional
            the base used for exponential transform. Defaults to natural
            exponent e.

        """
        super().__init__(**kwargs)
        self.base = base

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"base": self.base})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Exponential transform tf.Tensor.

        Parameters
        ----------
        inputs: tf.Tensor
            input tf.Tensor.

        Returns
        -------
        tf.Tensor
            exponential transformed tf.Tensor

        """
        outputs = tf.math.pow(self.base, inputs)

        return outputs
