from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cavachon.config.model_component_config import ModelComponentConfig
from cavachon.utils.general_utils import GeneralUtils


class ModelConfig(BaseModel):
    """ModelConfig

    Config mapping for model.

    Attributes
    ----------
    name: str
        name of the model.

    components: List[ComponentConfigMapping]
        list of component configs.

    load_weights: bool
        whether or not to load the pretrained weights before training.
    """

    name: str | None = Field(default="cavachon", description="name of the model.")
    components: List[ModelComponentConfig] = Field(
        description="list of component configs."
    )
    load_weights: bool = Field(
        default=False,
        description="whether or not to load the pretrained weights before training.",
    )
    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )

    @field_validator("name", mode="after")
    def convert_to_tensorflow_compatible_string(cls, value: str) -> str:
        """Convert to tensorflow compatible string.

        Parameters
        ----------
        value: str
            string to be converted.

        Returns
        -------
        str
            converted Tensorflow compatible string.

        """
        return GeneralUtils.convert_to_tensorflow_compatible_string(value)
