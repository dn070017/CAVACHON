import functools
from typing import Any, Mapping

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class OrderedLogisticDataModifier(tf.keras.Model):
    """Modifier applied right after the tf.data.Dataset is created.

    Densifies and shifts GISTIC2 states (-2..+2) to category indices
    (0..4), which is what OrderedLogistic.log_prob expects.
    """

    STATE_OFFSET = 2.0

    def __init__(self, modality_name):
        super().__init__()
        self.modality_name: str = modality_name
        self.modality_key: str = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs: Mapping[Any, tf.Tensor], training=None, mask=None):
        result = functools.reduce(
            lambda x, modifier: modifier(x), self.modifiers, inputs
        )
        result = dict(result)
        result[self.modality_key] = result[self.modality_key] + self.STATE_OFFSET
        return result
