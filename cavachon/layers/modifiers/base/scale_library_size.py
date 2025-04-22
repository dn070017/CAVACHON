from typing import Dict

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


class ScaleLibrarySize(tf.keras.layers.Layer):
    """ScaleLibrarySize

    Modifier used to scale up (read × library size) the tf.Tensor with
    library size (the sum of values in the last dimension).

    Attributes
    ----------
    key: str
        key to access the data needed to be normalized.
    """

    def __init__(self, key: str, show_warning: bool = True, *args, **kwargs):
        """Constructor for NormalizeLibrarySize

        Parameters
        ----------
        key: str
            key to access the data needed to be normalized.

        show_warning: bool, optional
            whether to show warning messages (with tf.print) when
            calling the layer. Defaults to True.

        """
        super().__init__(*args, **kwargs)
        self.key = key
        self.show_warning = show_warning

    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Scale tf.Tensor stored in input with library size (the sum
        of values in the last dimension)

        Parameters
        ----------
        inputs: Dict[str, tf.Tensor]
            inputs dictionary of tf.Tensor contains self.key

        Returns
        -------
        Dict[str, tf.Tensor]
            processed dictionary of tf.Tensor.

        """
        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        is_sparse = False
        tensor = outputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
            is_sparse = True

        libsize_key = f"{self.key}_{Constants.TENSOR_NAME_LIBSIZE}"

        if self.show_warning and libsize_key not in inputs:
            message = "".join(
                (
                    f"WARNING: {libsize_key} is not in batched data, ignore process in ",
                    f"{self.__class__.__name__}. Please use NormalizeLibrarySize ",
                    "in preprocessing.",
                )
            )
            tf.print(message)

        libsize = inputs.get(libsize_key, tf.ones((tf.shape(tensor)[0], 1)))
        tensor = tensor * libsize

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        outputs[self.key] = tensor
        outputs[libsize_key] = libsize
        return outputs
