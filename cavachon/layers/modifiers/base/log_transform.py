from typing import Dict

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


class LogTransform(tf.keras.layers.Layer):
    """LogTransform

    Modifier used to log-transform the tf.Tensor stored in a
    dictionary.

    Attributes
    ----------
    pseudocount: float
        values added to the tf.Tensor before log-transformation (to avoid
        -inf outcome)

    key: Hashable
        key to access the data needed to be log-transformed.

    """

    def __init__(self, key: str, pseudocount: float = 1.0, *args, **kwargs):
        """Constructor for LogTransform

        Parameters
        ----------
        key: str
            key to access the data needed to be binarized.

        pseudocount: float, optional
            values added to the tf.Tensor before log-transformation (to
            avoid -inf outcome)

        """
        super().__init__(*args, **kwargs)
        self.pseudocount = pseudocount
        self.key = key

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Log-transform tf.Tensor stored in inputs.

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

        tensor = tf.math.log(tensor + self.pseudocount)

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        outputs[self.key] = tensor
        return outputs
