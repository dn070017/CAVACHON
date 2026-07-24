from typing import List

from pydantic import model_validator

from cavachon.config.base import BaseConfigModel, TensorflowCompatibleStr
from cavachon.config.filter_config import FilterConfig
from cavachon.environment.constants import Constants


class ModalityConfig(BaseConfigModel):
    """Modality configuration model."""

    name: TensorflowCompatibleStr
    type: str
    dist: str = ""
    samples: List[str] = []
    h5ad: str = ""
    filters: List[FilterConfig] = []
    batch_effect_colnames: List[str] = []

    @model_validator(mode="after")
    def _set_defaults(self) -> "ModalityConfig":
        self.type = self.type.lower()
        if not self.dist:
            self.dist = Constants.DEFAULT_MODALITY_DISTRIBUTION.get(self.type, "")
        return self
