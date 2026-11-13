from typing import Any

from pydantic import field_validator

from cavachon.config.base import BaseConfigModel


class OptimizerConfig(BaseConfigModel):
    """Optimizer configuration model."""

    name: str = "adam"
    learning_rate: float = 1e-4

    @field_validator("learning_rate", mode="before")
    @classmethod
    def _coerce_learning_rate(cls, v: Any) -> float:
        # float() raises TypeError for e.g. None, which pydantic would not
        # convert into a ValidationError.
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"learning_rate should be a number, got {v!r}")
