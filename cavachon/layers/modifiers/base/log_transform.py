from typing import Hashable, MutableMapping

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


class LogTransform(tf.keras.layers.Layer):
    """LogTransform

    Modifier used to log-transform the tf.Tensor stored in a
    MutableMapping.

    Attributes
    ----------
    pseudocount: float
        values added to the tf.Tensor before log-transformation (to avoid
        -inf outcome)

    key: Hashable
        key to access the data needed to be log-transformed.

    """

    def __init__(self, key: Hashable, pseudocount: float = 1.0, *args, **kwargs):
        """Constructor for LogTransform

        Parameters
        ----------
        key: Hashable
            key to access the data needed to be binarized.

        pseudocount: float, optional
            values added to the tf.Tensor before log-transformation (to
            avoid -inf outcome)

        """
        super().__init__(*args, **kwargs)
        self.pseudocount = pseudocount
        self.key = key

    def call(
        self, inputs: MutableMapping[Hashable, tf.Tensor]
    ) -> MutableMapping[Hashable, tf.Tensor]:
        """Log-transform tf.Tensor stored in inputs.

        Parameters
        ----------
        inputs: MutableMapping[Hashable, tf.Tensor])
            inputs MutableMapping of tf.Tensor contains self.key

        Returns
        -------
        MutableMapping[Hashable, tf.Tensor]
            processed MutableMapping of tf.Tensor
        """
        is_sparse = False
        tensor = inputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
            is_sparse = True

        tensor = tf.math.log(tensor + self.pseudocount)

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        inputs[self.key] = tensor
        return inputs
