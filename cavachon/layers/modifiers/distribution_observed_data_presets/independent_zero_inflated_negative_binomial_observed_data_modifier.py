from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_preset_modifier import (
    DistributionPresetModifier,
)


class IndependentZeroInflatedNegativeBinomialObservedDataModifier(
    DistributionPresetModifier
):
    """IndependentZeroInflatedNegativeBinomialObservedDataModifier

    Modifiers for the modality which is
    IndependentZeroInflatedNegativeBinomial distribution. The instance
    will be used right after the tf.data.Dataset is created using the
    DataLoader.

    Attributes
    ----------
    modality_names: str
        modality name.

    modality_key: str
        the key used to access the mapping of data created from
        tf.data.Dataset. Defaults to `modality_name`_matrix_observed.

    modifiers: List[tf.keras.layers.Layer]
        list of modifiers that will be applied to the data created from
        tf.data.Dataset. Defaults to
        [ToDense, LogTransform, NormalizeLibrarySize].

    See Also
    --------
    DataLoader: used to create tf.data.Dataset from MuData.

    """

    def __init__(self, modality_name: str):
        """Constructor for IndependentZeroInflatedNegativeBinomial
        (modifier for tf.data.Dataset)

        Parameters
        ----------
        modality_name: str
            the name of modality that needs to be processed.
        """
        super().__init__()
        self.modality_name: str = modality_name
        self.modality_key: str = f"{modality_name}_{Constants.TENSOR_NAME_X_OBSERVED}"
        self.modifiers = [ToDense(self.modality_key)]
