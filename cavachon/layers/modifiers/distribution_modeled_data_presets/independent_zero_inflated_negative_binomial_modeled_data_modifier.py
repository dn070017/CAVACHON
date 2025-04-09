from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.log_transform import LogTransform
from cavachon.layers.modifiers.base.normalize_library_size import NormalizeLibrarySize
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)


class IndependentZeroInflatedNegativeBinomialModeledDataModifier(
    DistributionPresetModifier
):
    """IndependentZeroInflatedNegativeBinomialModeledDataModifier

    Modifiers for the modality which is
    IndependentZeroInflatedNegativeBinomial distribution. The instance
    will be used before calling the tf.keras.Model.

    Attributes
    ----------
    modality_names: str
        modality name.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Defaults to `modality_name`_matrix_modeled.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to
        [ToDense, LogTransform, NormalizeLibrarySize].

    See Also
    --------
    dataloader.modifiers.IndependentZeroInflatedNegativeBinomialModifier
        similar modifier but used for DataLoader.dataset.

    """

    def __init__(self, modality_name):
        """Constructor for IndependentZeroInflatedNegativeBinomial
        (modifier for tf.data.Dataset)

        Parameters
        ----------
        modality_name: str
            the name of modality that needs to be processed.
        """
        super().__init__()
        self.modality_name = modality_name
        self.modality_key = f"{modality_name}_{Constants.TENSOR_NAME_X_MODEL}"
        self.modifiers = [
            ToDense(self.modality_key),
            LogTransform(self.modality_key),
            NormalizeLibrarySize(self.modality_key),
        ]
