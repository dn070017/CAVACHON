from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cavachon.utils.general_utils import GeneralUtils


class ModelComponentConfig(BaseModel):
    """ModelComponentConfig

    Config mapping for component (in the model configuration).

    Attributes
    ----------
    name: str
        name of the component.

    conditioned_on_z: List[str]
        names of the conditioned components (of z).

    conditioned_on_z_hat: List[str]
        names of the conditioned components (of z_hat).

    modality_names: List[str]
        names of modalities used in inputs and outputs.

    distribution_names: Mapping[str, str]
        names of the distributions for each modality. The keys are the
        names of the modalities, and the values are the corresponding
        distribution names.

    n_latent_dims: int
        number of latent dimensions

    n_latent_priors: int
        number of priors for the latent distributions.

    n_encoder_layers: int
        number of encoder layers.

    n_decoder_layers: int
        number of decoder layers for each modality. The keys are the
        modality names, and the values are the corresponding number of
        decoder layers.

    n_progressive_epochs: int
        number of progressive epochs.

    """

    name: str = Field(description="name of the component.")
    modalities: List[str] = Field(
        description="modalities configured used in inputs and outputs"
    )
    conditioned_on_z: List[str] | None = Field(
        default_factory=list, description="names of the conditioned components (of z)."
    )
    conditioned_on_z_hat: List[str] | None = Field(
        default_factory=list,
        description="names of the conditioned components (of z_hat).",
    )
    n_latent_dims: int | None = Field(
        default=5, description="number of latent dimensions."
    )
    n_latent_priors: int | None = Field(
        default=11, description="number of priors for the latent distributions."
    )
    n_encoder_layers: int | None = Field(
        default=3, description="number of encoder layers."
    )
    n_decoder_layers: int | None = Field(
        default=3, description="number of encoder layers."
    )
    # TODO: consider move this to training config
    n_progressive_epochs: int | None = Field(
        default=1, description="number of progressive epochs in the training process."
    )

    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )

    @field_validator("name", mode="after")
    @classmethod
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

    @field_validator("modalities", mode="after")
    def convert_to_tensorflow_compatible_list_of_string(
        cls, value: List[str]
    ) -> List[str]:
        """Convert to tensorflow compatible list of strings.

        Parameters
        ----------
        value: List[str]
            strings to be converted.

        Returns
        -------
        List[str]
            converted Tensorflow compatible list of strings.

        """
        return [GeneralUtils.convert_to_tensorflow_compatible_string(v) for v in value]
