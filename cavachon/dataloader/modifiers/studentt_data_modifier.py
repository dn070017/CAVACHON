import functools
from typing import Any, Mapping

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class StudenttDataModifier(tf.keras.Model):
    def __init__(self, modality_name: str):
        super().__init__()
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs: Mapping[Any, tf.Tensor], **kwargs):
        return functools.reduce(lambda x, mod: mod(x), self.modifiers, inputs)
