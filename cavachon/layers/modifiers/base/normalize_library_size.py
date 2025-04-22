from typing import Dict

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


class NormalizeLibrarySize(tf.keras.layers.Layer):
    """NormalizedLibrarySize

    Modifier used to normalize (read / library size) the tf.Tensor with
    library size (the sum of values in the last dimension)

    Attributes
    ----------
    key: str
        key to access the data needed to be normalized.

    """

    def __init__(self, key: str, *args, **kwargs):
        """Constructor for NormalizeLibrarySize

        Parameters
        ----------
        key: str
            key to access the data needed to be normalized.

        """
        super().__init__(*args, **kwargs)
        self.key = key

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Normalize tf.Tensor stored in input with library size (the sum
        of values in the last dimension)

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor])
            inputs dictionary of tf.Tensor contains self.key

        Returns
        -------
        Dict[str, tf.Tensor]
            processed dictionary of tf.Tensor, where the library size
            used to normalize the data will be stored in
            (self.key, 'libsize'). This can be used to scale back to the
            original data.

        """
        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        is_sparse = False
        tensor = outputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
            is_sparse = True

        libsize_key = f"{self.key}_{Constants.TENSOR_NAME_LIBSIZE}"
        if libsize_key in outputs:
            libsize = outputs[libsize_key]
            tensor = tensor / libsize
        else:
            libsize = tf.expand_dims(tf.reduce_sum(tensor, axis=-1), -1)
            tensor = tensor / libsize
            outputs[libsize_key] = libsize

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)
        outputs[self.key] = tensor

        return outputs
