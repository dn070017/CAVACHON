from typing import Hashable, MutableMapping

import tensorflow as tf

from cavachon.utils.tensor_utils import TensorUtils


class ExpTransform(tf.keras.layers.Layer):
    """ExpTransform

    Modifier used to exponential transform the tf.Tensor stored in a
    MutableMapping.

    Attributes
    ----------
    key: Hashable
        key to access the data needed to be exponential transformed.

    """

    def __init__(self, key: Hashable, *args, **kwargs):
        """Constructor for ExpTransform

        Parameters
        ----------
        key: Hashable
            key to access the data needed to be exponential transformed.

        """
        super().__init__(*args, **kwargs)
        self.key = key

    def call(
        self, inputs: MutableMapping[Hashable, tf.Tensor]
    ) -> MutableMapping[Hashable, tf.Tensor]:
        """Exponential transform tf.Tensor stored in inputs.

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

        tensor = tf.math.exp(tensor)

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        inputs[self.key] = tensor
        return inputs
