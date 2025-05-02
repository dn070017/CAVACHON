from typing import Any, Dict

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


@tf.keras.utils.register_keras_serializable()
class Binarize(tf.keras.layers.Layer):
    """Binarize

    Modifier used to binarize the tf.Tensor stored in a dictionary.

    Attributes
    ----------
    key: str
        key to access the data needed to be binarized.

    threshold: float
        threshold used to binarize the tf.Tensor.

    """

    def __init__(self, key: str, threshold: float = 1.0, *args, **kwargs):
        """Constructor for Binarize

        Parameters
        ----------
        key: str
            key to access the data needed to be binarized.

        threshold: float, optional
            threshold used to binarize the data. If the value is larger
            than this threshold, it will be set to 1.0. Needs to be in the
            range of [0.0, 1.0] Defaults to 1.0.

        """
        super().__init__(*args, **kwargs)
        if threshold > 1.0 or threshold < 0.0:
            message = "".join(
                (
                    f"WARNING: invalid range for threshold. Expected 0 ≤ threshold ≤ 1, get {threshold}. Set to 1.0.",
                )
            )
            tf.print(message)
            threshold = 1.0
        self.threshold = threshold
        self.key = key

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            a dictionary containing the configuration of the layer.

        """
        config = super().get_config()
        config.update({"key": self.key, "threshold": self.threshold})
        return config

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor] | tf.Tensor:
        """Binarize to tf.Tensor stored in inputs.

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
        tensor = tf.where(
            tensor >= self.threshold, tf.ones_like(tensor), tf.zeros_like(tensor)
        )
        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        outputs[self.key] = tensor
        return outputs
