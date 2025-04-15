import warnings
from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizerParams(BaseModel):
    learning_rate: float | None = Field(
        default=0.001, description="learning rate of the optimizer."
    )
    use_ema: bool | None = Field(
        default=False,
        description="if True, exponential moving average (EMA) is applied.",
    )
    ema_momentum: float | None = Field(
        default=None,
        description="only used if use_ema=True. This is the momentum to use when computing the EMA of the model's weight",
    )
    ema_overwrite_frequency: int | None = Field(
        default=None,
        description="only used if use_ema=True. Every ema_overwrite_frequency steps of iterations, we overwrite the model variable by its moving average. If None, the optimizer does not overwrite model variables in the middle of training.",
    )
    weight_decay: float | None = Field(
        default=None, description="if set, weight decay is applied."
    )
    clipnorm: float | None = Field(
        default=None,
        description="if set, the gradient of each weight is individually clipped so that its norm is no higher than this value.",
    )
    clipvalue: float | None = Field(
        default=None,
        description="if set, the gradient of each weight is clipped to be no higher than this value.",
    )
    global_clipnorm: float | None = Field(
        default=None,
        description="if set, the gradient of all weights is clipped so that their global norm is no higher than this value.",
    )
    jit_compile: bool | None = Field(
        default=True, description="if True, the optimizer will use XLA compilation."
    )

    model_config = ConfigDict(revalidate_instances="always", validate_assignment=True)


class AdafactorOptimizerParams(OptimizerParams):
    name: Literal["adafactor"] = Field(
        default="adafactor",
        description="name of the optimizer. Should not be changed, used for discriminator in Pydantic.",
    )
    beta_2_decay: float | None = Field(
        default=-0.8, description="the decay rate of beta_2."
    )
    epsilon_1: float | None = Field(
        default=1e-30, description="a small offset to keep denominator away from 0."
    )
    epsilon_2: float | None = Field(
        default=0.001,
        description="a small offset to avoid learning rate becoming too small by time.",
    )
    clip_threshold: float | None = Field(default=1.0, description="clipping threshold")
    relative_step: bool | None = Field(
        default=True,
        description="if learning_rate is a constant and relative_step=True, learning rate will be adjusted based on current iterations.",
    )


class AdamOptimizerParams(OptimizerParams):
    name: Literal["adam"] = Field(
        default="adam",
        description="name of the optimizer. Should not be changed, used for discriminator in Pydantic.",
    )
    beta_1: float | None = Field(
        default=0.9,
        description="the exponential decay rate for the 1st moment estimates.",
    )
    beta_2: float | None = Field(
        default=0.999,
        description="the exponential decay rate for the 2nd moment estimates.",
    )
    epsilon: float | None = Field(
        default=1e-07, description="a small constant for numerical stability."
    )
    amsgrad: bool | None = Field(
        default=False, description="whether to apply AMSGrad variant of this algorithm"
    )


class AdamWOptimizerParams(AdamOptimizerParams):
    name: Literal["adamw"] = Field(  # type: ignore
        default="adamw",
        description="name of the optimizer. Should not be changed, used for discriminator in Pydantic.",
    )
    weight_decay: float | None = Field(
        default=0.004, description="if set, weight decay is applied."
    )


class LionOptimizerParams(OptimizerParams):
    name: Literal["lion"] = Field(
        default="lion",
        description="name of the optimizer. Should not be changed, used for discriminator in Pydantic.",
    )
    beta_1: float | None = Field(
        default=0.9,
        description="the rate to combine the current gradient and the 1st moment estimate.",
    )
    beta_2: float | None = Field(
        default=0.999,
        description="the exponential decay rate for the 1st moment estimate.",
    )


class TrainingOptimizerConfig(BaseModel):
    """TrainingOptimizerConfig

    Config for optimizer.

    Attributes
    ----------
    name: str
        name of the optimizer.

    params: AdafactorOptimizerParams | AdamOptimizerParams | AdamWOptimizerParams | LionOptimizerParams
        the parameters of the optimizer, see References.

    References
    ----------
    https://www.tensorflow.org/versions/r2.14/api_docs/python/tf/keras/optimizers: Tensorflow Keras optimizer documentation

    """

    kind: Literal[
        "adafactor",
        "adam",
        "adamw",
        "lion",
    ] = Field(default="adam", description="name of the optimizer.")
    params: (
        AdafactorOptimizerParams
        | AdamOptimizerParams
        | AdamWOptimizerParams
        | LionOptimizerParams
    ) = Field(..., discriminator="name", description="optimizer configuration.")

    model_config = ConfigDict(revalidate_instances="always", validate_assignment=True)

    @model_validator(mode="before")
    def set_default_params_based_on_name(cls, data: Any) -> Any:
        """Set default params based on the optimizer kind.

        If the 'params' field or 'params.name' is not provided in the
        input data, this validator sets a default 'params' dictionary
        containing only the 'name' key, derived from the 'kind' field.
        It also ensures the 'kind' field is lowercase.

        Parameters
        ----------
        data: Any
            the input data before validation.

        Returns
        -------
        Any
            the (potentially modified) data to be validated.
        """
        resulting_data = deepcopy(data)
        if isinstance(data, dict | BaseModel):
            optimizer_kind = data.get("kind", "adam")
            optimizer_kind = optimizer_kind.lower()
            resulting_data["kind"] = optimizer_kind

            if (
                "params" not in data
                or data["params"] is None
                or "name" not in data["params"]
            ):
                resulting_data["params"] = {"name": optimizer_kind}

        return resulting_data

    @model_validator(mode="after")
    def validate_name_consistency(self) -> Self:
        """Validate consistency between 'kind' and 'params.name'.

        Issues a warning if 'kind' and 'params.name' are both explicitly
        set to one of the allowed optimizer names ('adafactor', 'adam',
        'adamw', 'lion') but do not match each other.

        Returns
        -------
        TrainingOptimizerConfig
            The validated model instance.
        """
        if (
            (
                self.params is not None
                and self.params.name
                in [
                    "adafactor",
                    "adam",
                    "adamw",
                    "lion",
                ]
            )
            and (
                self.kind is not None
                and self.kind
                in [
                    "adafactor",
                    "adam",
                    "adamw",
                    "lion",
                ]
            )
            and (self.kind != self.params.name)
        ):
            warnings.warn(
                f"'{self.kind}' doesn't match with '{self.params.name}' when explicitly set to "
                "'adafactor', 'adam', 'adamw' and 'lion'. This may cause confusion for the "
                "configuration, the type of the optimizer will be created based "
                "on the params.name.",
                UserWarning,
            )
        return self
