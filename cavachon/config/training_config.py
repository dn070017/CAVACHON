from typing import Tuple, Union

from pydantic import model_validator, Field

from cavachon.config.base import BaseConfigModel
from cavachon.config.optimizer_config import OptimizerConfig


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
    """Training configuration model.

    The fields ``max_regular_training_epochs``, ``n_parent_annealing_epochs``,
    ``n_kl_annealing_epochs``, ``enable_kmeans_init``, and
    ``kl_annealing_ratio`` serve as global defaults. Each component can
    override them by setting its own value.
    """

    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(name="adam", learning_rate=1e-4))
    max_regular_training_epochs: int = 500
    n_parent_annealing_epochs: int = 1
    n_kl_annealing_epochs: int = 25
    enable_kmeans_init: bool = True
    kl_annealing_ratio: Tuple[float, float, float] = (0.5, 0.2, 0.3)
    train: bool = True
    early_stopping: Union[bool, EarlyStoppingConfig] = True

    @model_validator(mode="before")
    @classmethod
    def _default_optimizer(cls, data: object) -> object:
        if isinstance(data, dict) and "optimizer" not in data:
            data["optimizer"] = {"name": "adam", "learning_rate": 1e-4}
        return data
