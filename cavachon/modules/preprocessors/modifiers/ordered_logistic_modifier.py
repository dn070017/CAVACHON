import functools

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class OrderedLogisticModifier(tf.keras.Model):
    """Modifier applied before calling the tf.keras.Model.

    The category shift is applied in OrderedLogisticDataModifier, so
    only densification is needed here.
    """

    def __init__(self, modality_name):
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs, training=None, mask=None):
        return functools.reduce(lambda x, modifier: modifier(x), self.modifiers, inputs)
