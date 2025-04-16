from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cavachon.utils.general_utils import GeneralUtils


class DatasetModalityConfig(BaseModel):
    """DatasetModalityConfig

    Config for modality (inside dataset configuration)

    Attributes
    ----------
    name: str
        name of the modality.

    dist: str
        observed distribution of the modality.

    h5ad: str | None
        filename to the h5ad (if not provided with samples)

    batch_effect_colnames: List[str] | None
        the column names of the batch effects that needs to be
        corrected.
    """

    name: str = Field(description="name of the modality.")
    dist: Literal[
        "IndependentBernoulli",
        "IndependentZeroInflatedNegativeBinomial",
        "MultivariateNormalDiagDistribution",
    ] = Field(description="observed distribution of the modality.")
    h5ad: str | None = Field(
        default=None, description="filename to the h5ad (if not provided with samples)"
    )
    batch_effect_colnames: List[str] = Field(
        default_factory=list,
        description="the column names of the batch effects that needs to be corrected.",
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
