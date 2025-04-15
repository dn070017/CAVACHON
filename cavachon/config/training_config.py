import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from cavachon.config.training_optimizer_config import TrainingOptimizerConfig


class TrainingConfig(BaseModel):
    """TrainingConfig

    Config mapping for training.

    Attributes
    ----------
    optimizer: OptimizerConfig
        config for optimizer.

    max_n_epochs: int
        maximum number of epochs for training. Defaults to 500.

    train: bool
        whether or not to retrain (finetune) the model. Defaults to True.

    early_stopping: bool
        whether or not to use early stopping when training the model.
        Ignored if `train=False`. Defaults to True.

    save_weight: bool
        whether or not to save the weights after training.

    """

    optimizer: TrainingOptimizerConfig | None = Field(
        default_factory=TrainingOptimizerConfig, description="config for optimizer."
    )
    max_n_epochs: int | None = Field(
        default=500, description="maximum number of epochs for training."
    )
    train: bool | None = Field(
        default=True,
        description="whether or not to retrain (finetune) the model.",
        validate_default=True,
    )
    early_stopping: bool | None = Field(
        default=True,
        description="whether or not to use early stopping when training the model. Ignored if `train=False`.",
        validate_default=True,
    )
    save_weight: bool | None = Field(
        default=True,
        description="whether or not to save the weights after training.",
        validate_default=True,
    )

    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )

    @field_validator("train", mode="after")
    def validate_train(cls, value: bool, info: ValidationInfo) -> bool:
        """Validate train field.

        If train is False, check if early_stopping or save_weight is
        True and issue a warning if so, as these fields are ignored
        when train is False.

        Parameters
        ----------
        value: bool
            value of the field.

        info: ValidationInfo
            validation info from pydantic.

        Returns
        -------
        bool
            validated value of the field.
        """
        if isinstance(value, bool) and not value:
            for field in ["early_stopping", "save_weight"]:
                field_value = info.data.get(field, None)
                if (
                    field_value is not None
                    and isinstance(field_value, bool)
                    and field_value
                ):
                    warnings.warn(
                        f"{info.data} is ignored when train=False.",
                        UserWarning,
                    )

        return value

    @field_validator("early_stopping", "save_weight", mode="after")
    def validate_early_stopping_save_weight(
        cls, value: bool, info: ValidationInfo
    ) -> bool:
        """Validate early_stopping and save_weight fields.

        If train is False, check if the field (early_stopping or
        save_weight) is True and issue a warning if so, as these fields
        are ignored when train is False.

        Parameters
        ----------
        value: bool
            value of the field.

        info: ValidationInfo
            validation info from pydantic.

        Returns
        -------
        bool
            validated value of the field.
        """
        train = info.data.get("train", None)
        train_is_false = train is not None and isinstance(train, bool) and not train
        if train_is_false and value:
            warnings.warn(
                f"{info.field_name} is ignored when train=False.", UserWarning
            )

        return value
