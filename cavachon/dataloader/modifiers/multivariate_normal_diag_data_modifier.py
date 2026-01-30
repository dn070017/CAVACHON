import functools
from typing import Any, Mapping

import tensorflow as tf

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.to_dense import ToDense


class MultivariateNormalDiagDataModifier(tf.keras.Model):
    """MultivariateNormalDiagDataModifier

    Modifiers for the modality which follows a MultivariateNormalDiag 
    distribution (Normal distribution with diagonal covariance).
    The instance will be used right after the tf.data.Dataset is 
    created using the DataLoader.

    Attributes
    ----------
    modality_names: str
        modality name.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Defaults to `modality_name`_matrix.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to [ToDense].

    See Also
    --------
    DataLoader: used to create tf.data.Dataset from MuData.

    """

    def __init__(self, modality_name: str):
        """Constructor for MultivariateNormalDiag data modifier

        Parameters
        ----------
        modality_name: str
            the name of modality that needs to be processed.
        """
        super().__init__()
        self.modality_name: str = modality_name
        self.modality_key: str = f"{modality_name}_{Constants.TENSOR_NAME_X}"
        # For continuous normalized data (CNV, normalized RNA, etc.)
        # we only need to convert sparse matrices to dense tensors
        self.modifiers = [ToDense(self.modality_key)]

    def call(self, inputs: Mapping[Any, tf.Tensor], training=None, mask=None):
        """Process the data created from tf.data.Dataset.

        Parameters
        ----------
        inputs:
            Mapping of tf.Tensor, where the keys contain
            self.modality_key.

        training: bool, optional
            Not used (kept for tf.keras.Model API).

        mask: tf.Tensor, optional
            Not used (kept for tf.keras.Model API).

        Returns
        -------
        Mapping[Any, tf.Tensor]
            processed data.

        """
        modifiers = self.modifiers
        return functools.reduce(lambda x, modifier: modifier(x), modifiers, inputs)