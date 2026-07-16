from typing import Union

from pydantic import model_validator, Field

from cavachon.config.models.base import BaseConfigModel
from cavachon.config.models.optimizer_config import OptimizerConfig


class EarlyStoppingConfig(BaseConfigModel):
    """Early stopping configuration model.

    Note: the scheduler currently constructs the callback with its own
    hard-coded parameters, so these values are accepted and stored but
    not consumed by the training loop yet.
    """

    monitor: str = "loss"
    mode: str = "min"
    patience: int = 25


class TrainingConfig(BaseConfigModel):
    """Training configuration model."""

    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(name="adam", learning_rate=1e-4))
    max_regular_training_epochs: int = 500
    train: bool = True
    early_stopping: Union[bool, EarlyStoppingConfig] = True

    @model_validator(mode="before")
    @classmethod
    def _default_optimizer(cls, data: object) -> object:
        if isinstance(data, dict) and "optimizer" not in data:
            data["optimizer"] = {"name": "adam", "learning_rate": 1e-4}
        return data
