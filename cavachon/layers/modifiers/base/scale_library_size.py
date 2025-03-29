import warnings
from typing import Hashable, Iterable, MutableMapping

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


class ScaleLibrarySize(tf.keras.layers.Layer):
    """ScaleLibrarySize

    Modifier used to scale (read × library size) the tf.Tensor with
    library size (the sum of values in the last dimension).

    Attributes
    ----------
    key: Hashable
        key to access the data needed to be normalized.
    """

    def __init__(self, key: Hashable, *args, **kwargs):
        """Constructor for NormalizeLibrarySize

        Parameters
        ----------
        key: Hashable
            key to access the data needed to be normalized.

        """
        super().__init__(*args, **kwargs)
        self.show_warning = True
        self.key = key

    def call(
        self, inputs: MutableMapping[str, tf.Tensor]
    ) -> MutableMapping[str, tf.Tensor]:
        """Scale tf.Tensor stored in input with library size (the sum
        of values in the last dimension)

        Parameters
        ----------
        inputs: MutableMapping[Hashable, tf.Tensor])
            inputs MutableMapping of tf.Tensor contains self.key

        Returns
        -------
        MutableMapping[Hashable, tf.Tensor]
            processed MutableMapping of tf.Tensor.

        """
        is_sparse = False
        tensor = inputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
            is_sparse = True

        if isinstance(self.key, str):
            libsize_key = f"{self.key}_{Constants.TENSOR_NAME_LIBSIZE}"
        elif isinstance(self.key, Iterable):
            libsize_key = self.key[:-1] + (Constants.TENSOR_NAME_LIBSIZE,)
        else:
            raise NotImplementedError(
                f"Expected key to be str or a hashable Iterable, get {type(self.key)}"
            )

        if self.show_warning and libsize_key not in inputs:
            message = "".join(
                (
                    f"{libsize_key} is not in batched data, ignore process in ",
                    f"{self.__class__.__name__}. Please use NormalizeLibrarySize ",
                    "in preprocessing.",
                )
            )
            warnings.warn(message, RuntimeWarning)
            self.show_warning = False

        libsize = inputs.get(libsize_key, tf.ones((tensor.shape[0], 1)))
        tensor = tensor * libsize

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        inputs[self.key] = tensor
        inputs[libsize_key] = libsize
        return inputs
