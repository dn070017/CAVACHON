import functools

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.binarize import Binarize
from cavachon.layers.modifiers.base.to_dense import ToDense


class IndependentBernoulliModifier(tf.keras.layers.Layer):
    """IndependentBernoulliModifier

    Modifiers for the modality which is IndependentBernoulli
    distribution. The instance will be used before calling the
    tf.keras.Model. Note that this will not change the data in the
    DataLoader.dataset which
    dataloader.modifiers.IndependentBernoulliModifier does.

    Attributes
    ----------
    modality_names: str
        modality name.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Defaults to `modality_name`_matrix.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to [ToDense, Binarize].

    See Also
    --------
    dataloader.modifiers.IndependentBernoulliModifier
        similar modifier but used for DataLoader.dataset.

    """

    def __init__(self, modality_name):
        """Constructor for IndependentBernoulli (modifier for
        tf.data.Dataset)

        Parameters
        ----------
        modality_name: str
            the name of modality that needs to be processed.

        """
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        self.modifiers = [ToDense(self.modality_key), Binarize(self.modality_key)]

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
