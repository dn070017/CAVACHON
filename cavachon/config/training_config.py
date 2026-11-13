from typing import Optional, Tuple, Union, overload

from pydantic import Field, field_validator, model_validator

from cavachon.config.base import BaseConfigModel
from cavachon.config.optimizer_config import OptimizerConfig


@overload
def validate_kl_annealing_ratio(
    v: Tuple[float, float, float],
) -> Tuple[float, float, float]: ...


@overload
def validate_kl_annealing_ratio(v: None) -> None: ...


def validate_kl_annealing_ratio(
    v: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    """Validate kl_annealing_ratio: the fractions of the KL annealing
    epochs spent on (1) standard KL only, (2) the crossfade from standard
    KL to GMM KL and (3) GMM KL only. They must be non-negative and sum
    to 1."""
    if v is None:
        return v
    if any(r < 0 for r in v):
        raise ValueError(f"kl_annealing_ratio should be non-negative, got {v}")
    if abs(sum(v) - 1.0) > 1e-6:
        raise ValueError(f"kl_annealing_ratio should sum to 1, got {v} (sum {sum(v)})")
    return v


class EarlyStoppingConfig(BaseConfigModel):
    """Early stopping configuration model.

    Used by the scheduler for the regular (GMM) training phase and the
    z_hat training phase. ``early_stopping: true`` in the training
    configuration uses these defaults. The best weights are always
    restored when training stops early.

    Attributes
    ----------
    monitor: str
        the logged quantity to monitor, e.g. 'loss'. Defaults to 'loss'.

    mode: str
        one of 'min', 'max' or 'auto'. Defaults to 'min'.

    patience: int
        number of epochs without improvement after which training is
        stopped. Defaults to 25.

    min_delta: float
        minimum change of the monitored quantity (in its own units) that
        counts as an improvement. Defaults to 5.0.
    """

    monitor: str = "loss"
    mode: str = "min"
    patience: int = 25
    min_delta: float = 5.0

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("min", "max", "auto"):
            raise ValueError(f"mode should be one of 'min', 'max' or 'auto', got {v!r}")
        return v

    @field_validator("patience")
    @classmethod
    def _validate_patience(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"patience should be a non-negative integer, got {v}")
        return v

    @field_validator("min_delta")
    @classmethod
    def _validate_min_delta(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"min_delta should be non-negative, got {v}")
        return v


class TrainingConfig(BaseConfigModel):
    """Training configuration model.

    The fields ``max_regular_training_epochs``, ``n_parent_annealing_epochs``,
    ``n_kl_annealing_epochs``, ``enable_kmeans_init``, ``learn_z_hat_priors``,
    ``kl_annealing_ratio``, ``n_z_hat_annealing_epochs``,
    ``n_z_hat_training_epochs`` and ``z_hat_kl_weight`` serve as global
    defaults. Each component can override them by setting its own value.

    The z_hat annealing phase runs after the regular training of a
    component with ``learn_z_hat_priors`` enabled: the encoder is frozen,
    z_hat becomes stochastic and its GMM KL weight is ramped from 0 to
    ``z_hat_kl_weight`` over ``n_z_hat_annealing_epochs``, then kept for
    ``n_z_hat_training_epochs``. Setting ``n_z_hat_annealing_epochs`` to
    0 disables the phase. Root components (no parents) skip it: their
    z_hat is z and the z prior serves as the z_hat prior, so
    ``learn_z_hat_priors`` is ignored for them.
    """

    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(name="adam", learning_rate=1e-4)
    )
    max_regular_training_epochs: int = Field(default=500, ge=0)
    n_parent_annealing_epochs: int = Field(default=1, ge=0)
    n_kl_annealing_epochs: int = Field(default=25, ge=0)
    enable_kmeans_init: bool = True
    learn_z_hat_priors: bool = False
    kl_annealing_ratio: Tuple[float, float, float] = (0.5, 0.2, 0.3)
    n_z_hat_annealing_epochs: int = Field(default=0, ge=0)
    n_z_hat_training_epochs: int = Field(default=0, ge=0)
    z_hat_kl_weight: float = Field(default=1.0, ge=0)
    train: bool = True
    early_stopping: Union[bool, EarlyStoppingConfig] = True

    @field_validator("kl_annealing_ratio")
    @classmethod
    def _validate_kl_annealing_ratio(
        cls, v: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        return validate_kl_annealing_ratio(v)

    @model_validator(mode="before")
    @classmethod
    def _default_optimizer(cls, data: object) -> object:
        if isinstance(data, dict) and "optimizer" not in data:
            # copy so the caller's dict is not mutated.
            data = {**data, "optimizer": {"name": "adam", "learning_rate": 1e-4}}
        return data
