from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.config.config_mapping.modality_file_feature_config_mapping import (
    ModalityFileFeatureConfigMapping,
)
from cavachon.config.config_mapping.modality_file_matrix_config_mapping import (
    ModalityFileMatrixConfigMapping,
)
from cavachon.utils.GeneralUtils import GeneralUtils


class ModalityFileConfigMapping(ConfigMapping):
    """ModalityFileConfigMapping

    Config mapping for modality.

    Attributes
    ----------
    name: str
        name of the modality.

    matrix: str
        config for the matrix file.

    barcodes: str
        config for the barcodes file.

    features: str
        config for the features file.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for ModalityFileConfigMapping

        Parameters
        ----------
        name: str
            name of the modality.

        matrix: MutableMapping[str, Any]
            config for the matrix file in MutableMapping format.

        barcodes: MutableMapping[str, Any]
            config for the barcodes file in MutableMapping format.

        features: MutableMapping[str, Any]
            config for the features file in MutableMapping format.

        """
        self.name: str
        self.matrix: ModalityFileMatrixConfigMapping
        self.barcodes: ModalityFileFeatureConfigMapping
        self.features: ModalityFileFeatureConfigMapping

        super().__init__(kwargs, ["name", "matrix", "barcodes", "features"])

        # postprocessing
        self.name = GeneralUtils.tensorflow_compatible_str(self.name)
        self.matrix = ModalityFileMatrixConfigMapping(**self.matrix)
        self.barcodes = ModalityFileFeatureConfigMapping(**self.barcodes)
        self.features = ModalityFileFeatureConfigMapping(**self.features)
