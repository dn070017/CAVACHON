from typing import Hashable, MutableMapping

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


class ToSparse(tf.keras.layers.Layer):
    """ToDense

    Modifier used to convert tf.Tensor stored in a MutableMapping to
    tf.SparseTensor.

    Attributes
    ----------
    key: Hashable
        key to access the data needed to be transform to tf.SparseTensor.

    """

    def __init__(self, key: Hashable, *args, **kwargs):
        """Constructor for ToSparse

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
        """Transform tf.Tensor stored in inputs to tf.SparseTensor.

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
        if isinstance(
            tensor, (tf.keras.KerasTensor, tf.SparseTensor)
        ) and not TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.from_dense(tensor)

        inputs[self.key] = tensor
        return inputs
