import functools

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
        """Constructor for IndependentBernoulli (modifier for
        tf.data.Dataset)

        Parameters
        ----------
        modality_name: str
            the name of modality that needs to be processed.

        """
        super().__init__()
        self.modality_name = ""
        self.modality_key = ""
        self.modifiers = []

    def call(self, inputs):
        """Processed the data created from tf.data.Dataset.

        Parameters
        ----------
        inputs:
            mapping of tf.Tensor, where the keys contain
            self.modality_key.

        Returns
        -------
        Mapping[Any, tf.Tensor]
            processed data.

        """
        modifiers = self.modifiers
        return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)
