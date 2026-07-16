from typing import List

from cavachon.config.base import BaseConfigModel
from cavachon.config.modality_file_config import ModalityFileConfig


class SampleConfig(BaseConfigModel):
    """Sample configuration with associated modality files."""

    name: str
    description: str = ""
    modalities: List[ModalityFileConfig] = []
