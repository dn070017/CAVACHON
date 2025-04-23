from typing import Dict

import tensorflow as tf


class DistributionPresetModifier(tf.keras.layers.Layer):
    """DistributionPresetModifier

    Parent class for distribution preset modifiers.

    Attributes
    ----------
    modality_names: str
        modality name. Please overwrite this in child classes.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Please overwrite this in child classes.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Please overwrite this in child classes.

    See Also
    --------
    dataloader.modifiers.IndependentBernoulliModifier
        similar modifier but used for DataLoader.dataset.

    """

    def __init__(self):
        """Constructor for distribution preset modifiers"""
        super().__init__()
        self.modality_name = ""
        self.modality_key = ""
        self.modifiers = []

    def call(self, inputs: Dict[str, tf.Tensor]):
        """Processed the data created from tf.data.Dataset.

        Parameters
        ----------
        inputs:
            mapping of tf.Tensor, where the keys contain
            self.modality_key.

        Returns
        -------
        Dict[str, tf.Tensor]
            processed data.

        """
        outputs = {k: tf.identity(v) for k, v in inputs.items()}
        for modifier in self.modifiers:
            outputs = modifier(outputs)
        return outputs
