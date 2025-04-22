from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)


class MultivariateNormalDiagModeledDataModifier(DistributionPresetModifier):
    """MultivariateNormalDiagModeledDataModifier

    Modifiers for the modality which is MultivariateNormalDiag
    distribution. The instance will be used before calling the
    tf.keras.Model.

    Attributes
    ----------
    modality_names: str
        modality name.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Defaults to `modality_name`_matrix_modeled.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to [ToDense, Binarize].

    See Also
    --------
    dataloader.modifiers.MultivariateNormalDiagModifier
        similar modifier but used for DataLoader.dataset.

    """

    def __init__(self, modality_name: str):
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X_MODEL}"
        self.modifiers = [ToDense(self.modality_key)]
