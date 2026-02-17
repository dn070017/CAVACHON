import functools

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class StudenttModifier(tf.keras.Model):
    """
    Modifier for Student-T modalities used during the model's preprocessing
    step. This ensures that the data is converted to a Dense tensor
    immediately before the model call.
    """

    def __init__(self, modality_name: str):
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        # For CNV data, we just need the ToDense conversion.
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs, training=None, mask=None):
        """
        Applies the sequence of modifiers to the input mapping.
        """
        modifiers = self.modifiers
        return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)
