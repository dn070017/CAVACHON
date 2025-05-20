import warnings
from typing import Any, Dict

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Binarize(tf.keras.layers.Layer):
    """Binarize

    Modifier used to binarize the tf.Tensor.

    Attributes
    ----------
    threshold: float
        threshold used to binarize the tf.Tensor.

    """

    def __init__(
        self,
        threshold: float = 1.0,
        **kwargs,
    ):
        """Constructor for Binarize

        Parameters
        ----------
        threshold: float, optional
            threshold used to binarize the data. If the value is larger
            than this threshold, it will be set to 1.0. Needs to be in the
            range of [0.0, 1.0] Defaults to 1.0.

        """
        super().__init__(**kwargs)
        if threshold > 1.0 or threshold < 0.0:
            clip_threshold = np.clip(threshold, 0.0, 1.0)
            message = (
                f"invalid range for threshold. Expected 0 ≤ threshold ≤ 1"
                f", get {threshold}. Set to {clip_threshold}"
            )
            warnings.warn(message, UserWarning)
            threshold = clip_threshold

        self.threshold = threshold

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Binarize to tf.Tensor.

        Parameters
        ----------
        inputs: tf.Tensor
            input tf.Tensor.

        Returns
        -------
        tf.Tensor
            binarized tf.Tensor

        """
        outputs = tf.where(
            inputs >= self.threshold, tf.ones_like(inputs), tf.zeros_like(inputs)
        )

        return outputs
