from typing import Hashable, MutableMapping

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


class ToDense(tf.keras.layers.Layer):
    """ToDense

    Modifier used to convert tf.SparseTensor stored in a MutableMapping
    to dense tf.Tensor.

    Attributes
    ----------
    key: Hashable
        key to access the data needed to be transform to dense
        tf.Tensor.

    """

    def __init__(self, key: Hashable, *args, **kwargs):
        """Constructor for ToDense

        Parameters
        ----------
        key: Hashable
            key to access the data needed to be transformed.

        """
        super().__init__(*args, **kwargs)
        self.key = key

    def call(
        self, inputs: MutableMapping[str, tf.Tensor]
    ) -> MutableMapping[str, tf.Tensor]:
        """Transform tf.SparseTensor stored in inputs to tf.Tensor.

        Parameters
        ----------
        inputs: MutableMapping[Hashable, tf.Tensor])
            inputs MutableMapping of tf.Tensor contains self.key

        Returns
        -------
        MutableMapping[Hashable, tf.Tensor]
            processed MutableMapping of tf.Tensor

        """

        tensor = inputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
        inputs[self.key] = tensor
        return inputs
