from typing import Any, Dict

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class ToSparse(tf.keras.layers.Layer):
    """ToDense

    Modifier used to convert tf.Tensor stored in a dictionary to
    tf.SparseTensor.

    Attributes
    ----------
    key: str
        key to access the data needed to be transform to tf.SparseTensor.

    """

    def __init__(self, key: str, *args, **kwargs):
        """Constructor for ToSparse

        Parameters
        ----------
        key: str
            key to access the data needed to be transformed.

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
        """Transform tf.Tensor stored in inputs to tf.SparseTensor.

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor])
            inputs dictionary of tf.Tensor contains self.key

        Returns
        -------
        Dict[str, tf.Tensor]
            processed dictionary of tf.Tensor

        """
        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        tensor = outputs[self.key]
        if isinstance(
            tensor, (tf.keras.KerasTensor, tf.Tensor)
        ) and not TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.from_dense(tensor)

        outputs[self.key] = tensor
        return outputs
