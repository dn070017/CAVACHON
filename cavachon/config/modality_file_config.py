from typing import List

from cavachon.config.base import BaseConfigModel, TensorflowCompatibleStr


class ModalityFileMatrixConfig(BaseConfigModel):
    """Configuration for a modality matrix file."""

    filename: str
    transpose: bool = False


class ModalityFileFeatureConfig(BaseConfigModel):
    """Configuration for modality feature (obs / var) files."""

    filename: str
    has_headers: bool = False
    colnames: List[str] = []


class ModalityFileConfig(BaseConfigModel):
    """Per-sample modality file configuration."""

    name: TensorflowCompatibleStr
    matrix: ModalityFileMatrixConfig
    barcodes: ModalityFileFeatureConfig
    features: ModalityFileFeatureConfig
