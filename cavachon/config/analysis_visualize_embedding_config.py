from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cavachon.utils.general_utils import GeneralUtils


class AnalysisVisualizeEmbeddingConfig(BaseModel):
    """AnalysisVisualizeEmbeddingConfig

    Config for embedding visualization using Pydantic.

    Attributes
    ----------
    modality: str
        which modality to visualize. It will be transformed to be
        TensorFlow compatible.

    use_rep: str
        which representation of the modality's obs to visualize.

    embedding_method: Literal['pca', 'umap', 'tsne']
        method used to embed the representation of the modality.
        Defaults to 'tsne'.

    color_by: str
        color by which annotation column in the modality's obs.

    interactive: bool
        whether or not to create interactive visualization. Defaults to
        False.
    """

    modality: str = Field(
        description="which modality to visualize. It will be transformed to a TensorFlow compatible string"
    )
    use_rep: str = Field(
        description="which representation of the modality to visualize (e.g., a component name)."
    )
    embedding_method: Literal["pca", "umap", "tsne"] = Field(
        default="tsne",
        description="method used to embed the representation of the modality.",
    )
    color_by: str = Field(
        description="color by which annotation column in the modality's obs."
    )
    interactive: bool = Field(
        default=False,
        description="whether or not to create interactive visualization.",
    )

    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )

    @field_validator("modality", mode="after")
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
