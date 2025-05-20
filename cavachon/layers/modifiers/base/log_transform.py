import math
import warnings
from typing import Any, Dict

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class LogTransform(tf.keras.layers.Layer):
    """LogTransform

    Modifier used to log-transform the tf.Tensor.

    Attributes
    ----------
    base: float, optional
        the base used for log transform.

    pseudocount: float
        values added to the tf.Tensor before log-transformation (to
        avoid -inf outcome)

    """

    def __init__(
        self,
        base: float = math.e,
        pseudocount: float = 1.0,
        **kwargs,
    ):
        """Constructor for LogTransform

        Parameters
        ----------
        base: float, optional
            the base used for logarithmic transform. Defaults to natural
            exponent e.

        pseudocount: float, optional
            values added to the tf.Tensor before log-transformation (to
            avoid -inf outcome)

        """
        super().__init__(**kwargs)
        if base < 0.0:
            message = (
                f"invalid range for base. Expected 0 ≤ base, "
                f"get {base}. Set to natural exponent e"
            )
            warnings.warn(message, UserWarning)
            base = math.e

        if pseudocount < 0.0:
            message = (
                f"invalid range for pseudocount. Expected 0 ≤ "
                f"pseudocount, get {pseudocount}. Set to 1.0"
            )
            warnings.warn(message, UserWarning)
            pseudocount = 1.0

        self.base = base
        self.pseudocount = pseudocount

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"pseudocount": self.pseudocount, "base": self.base})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Logarithmic transform tf.Tensor.

        Parameters
        ----------
        inputs: tf.Tensor
            input tf.Tensor.

        Returns
        -------
        tf.Tensor
            logarithmic transformed tf.Tensor
        """
        outputs = tf.math.log(inputs + self.pseudocount) / tf.math.log(self.base)

        return outputs
