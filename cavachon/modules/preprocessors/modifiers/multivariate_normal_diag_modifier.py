import functools

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class MultivariateNormalDiagModifier(tf.keras.Model):
    """MultivariateNormalDiagModifier

    Modifiers for the modality which is MultivariateNormalDiag
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
        tf.data.Dataset. Defaults to `modality_name`/matrix.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to [ToDense, Binarize].

    See Also
    --------
    dataloader.modifiers.MultivariateNormalDiagModifier
        similar modifier but used for DataLoader.dataset.

    """

    def __init__(self, modality_name):
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}/{Constants.TENSOR_NAME_X}"
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs, training=None, mask=None):
        """Processed the data created from tf.data.Dataset.

        Parameters
        ----------
        inputs:
            mapping of tf.Tensor, where the keys contain
            self.modality_key.

        training: bool, optional
            not used (kept for tf.keras.Model API).

        mask: tf.Tensor, optional
            not used (kept for tf.keras.Model API).

        Returns
        -------
        Mapping[Any, tf.Tensor]
            processed data.

        """
        modifiers = self.modifiers
        return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)
