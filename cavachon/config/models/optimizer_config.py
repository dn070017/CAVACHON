from pydantic import field_validator

from cavachon.config.models.base import BaseConfigModel


class OptimizerConfig(BaseConfigModel):
    """Optimizer configuration model."""

    name: str = "adam"
    learning_rate: float = 1e-4

    @field_validator("learning_rate", mode="before")
    @classmethod
    def _coerce_learning_rate(cls, v: object) -> float:
        return float(v)
