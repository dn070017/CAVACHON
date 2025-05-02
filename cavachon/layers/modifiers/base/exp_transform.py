from typing import Any, Dict

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class ExpTransform(tf.keras.layers.Layer):
    """ExpTransform

    Modifier used to exponential transform the tf.Tensor stored in a
    dictionary.

    Attributes
    ----------
    key: str
        key to access the data needed to be exponential transformed.

    """

    def __init__(self, key: str, *args, **kwargs):
        """Constructor for ExpTransform

        Parameters
        ----------
        key: str
            key to access the data needed to be exponential transformed.

        """
        super().__init__(*args, **kwargs)
        self.key = key

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"key": self.key})
        return config

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Exponential transform tf.Tensor stored in inputs.

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            inputs dictionary of tf.Tensor contains self.key

        Returns
        -------
        Dict[str, tf.Tensor]
            processed dictionary of tf.Tensor
        """
        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        is_sparse = False
        tensor = outputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)

            is_sparse = True

        tensor = tf.math.exp(tensor)

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        outputs[self.key] = tensor
        return outputs
