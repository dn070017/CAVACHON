from typing import List

from cavachon.config.models.base import BaseConfigModel
from cavachon.config.models.modality_file_config import ModalityFileConfig


class SampleConfig(BaseConfigModel):
    """Sample configuration with associated modality files."""

    name: str
    description: str = ""
    modalities: List[ModalityFileConfig] = []
