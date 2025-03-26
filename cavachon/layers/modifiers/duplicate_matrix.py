from typing import Hashable, Iterable, MutableMapping

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.utils.tensor_utils import TensorUtils


class DuplicateMatrix(tf.keras.layers.Layer):
    """DuplicateMatrix

    Modifier used to duplicate the tf.Tensor stored in a MutableMapping.

    Attributes
    ----------
    key: Any
        key to access the data needed to be duplicated.

    threshold: float
        threshold used to duplicate the tf.Tensor.

    """

    def __init__(self, key: Hashable, *args, **kwargs):
        """Constructor for DuplicateMatrix

        Parameters
        ----------
        key: Hashable
            key to access the data needed to be duplicated.

        """
        super().__init__(*args, **kwargs)
        self.key = key

    def call(
        self, inputs: MutableMapping[Hashable, tf.Tensor]
    ) -> MutableMapping[Hashable, tf.Tensor]:
        """Duplicate the tf.Tensor stored in inputs.

        Parameters
        ----------
        inputs : MutableMapping[Hashable, tf.Tensor]
            inputs MutableMapping of tf.Tensor contains self.key

        Returns
        -------
        MutableMapping[Hashable, tf.Tensor]
            processed MutableMapping of tf.Tensor.

        Raises
        ------
        NotImplementedError
            if self.key is not str or a hashable Iterable.
        """
        is_sparse = False
        tensor = inputs[self.key]
        if TensorUtils.is_sparse_tensor(tensor):
            tensor = tf.sparse.to_dense(tensor)
            is_sparse = True

        if isinstance(self.key, str):
            duplicate_key = f"{self.key}_{Constants.TENSOR_NAME_ORIGIONAL_X}"
        elif isinstance(self.key, Iterable):
            duplicate_key = self.key[:-1] + (Constants.TENSOR_NAME_ORIGIONAL_X,)
        else:
            raise NotImplementedError(
                f"Expected key to be str or a hashable Iterable, get {type(self.key)}"
            )

        if is_sparse:
            tensor = tf.sparse.from_dense(tensor)

        inputs[duplicate_key] = tensor
        return inputs
