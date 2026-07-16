from typing import List

from pydantic import Field

from cavachon.config.base import BaseConfigModel, TensorflowCompatibleStr
from cavachon.config.component_config import ComponentConfig
from cavachon.config.dataset_config import DatasetConfig
from cavachon.config.training_config import TrainingConfig


class ModelConfig(BaseConfigModel):
    """Model configuration model."""

    name: TensorflowCompatibleStr = "cavachon"
    components: List[ComponentConfig] = []
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    load_weights: bool = False
    save_weights: bool = True
