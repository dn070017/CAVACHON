from pydantic import BaseModel, ConfigDict, Field, field_validator

from cavachon.utils.general_utils import GeneralUtils


class AnalysisGenericConfig(BaseModel):
    """AnalysisGenericConfig

    Config for analysis which uses modality and component.

    Attributes
    ----------
    modality: str
        which modality of the outputs of the component to use. It
        will be transformed to be TensorFlow compatible.

    component: str
        the outputs of which component to use. It will be transformed
        to be TensorFlow compatible.

    """

    modality: str = Field(
        description="which modality of the outputs of the component to use. It will be transformed to be TensorFlow compatible."
    )
    component: str = Field(
        description="the outputs of which component to use. It will be transformed to be TensorFlow compatible."
    )
    model_config = ConfigDict(revalidate_instances="always", validate_assignment=True)

    @field_validator("modality", "component", mode="after")
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
