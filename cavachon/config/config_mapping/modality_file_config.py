from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.config.config_mapping.modality_file_feature_config import (
    ModalityFileFeatureConfig,
)
from cavachon.config.config_mapping.modality_file_matrix_config import (
    ModalityFileMatrixConfig,
)
from cavachon.utils.GeneralUtils import GeneralUtils


class ModalityFileConfig(ConfigMapping):
    """ModalityFileConfig

    Config for modality.

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
        """Constructor for ModalityFileConfig

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
        self.matrix: ModalityFileMatrixConfig
        self.barcodes: ModalityFileFeatureConfig
        self.features: ModalityFileFeatureConfig

        super().__init__(kwargs, ["name", "matrix", "barcodes", "features"])

        # postprocessing
        self.name = GeneralUtils.tensorflow_compatible_str(self.name)
        self.matrix = ModalityFileMatrixConfig(**self.matrix)
        self.barcodes = ModalityFileFeatureConfig(**self.barcodes)
        self.features = ModalityFileFeatureConfig(**self.features)
