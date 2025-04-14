from typing import List

from pydantic import Field, field_validator

from cavachon.config.analysis_generic_config import AnalysisGenericConfig
from cavachon.utils.general_utils import GeneralUtils


class AnalysisAttributionScoreConfig(AnalysisGenericConfig):
    """AnalysisAttributionScoreConfig

    Config for attribution score analysis. Inherits modality and component
    from AnalysisGenericConfig.

    Attributes
    ----------
    with_respect_to: List[str]
        list of component names to compute integrated gradients with respect to
        their latent representations. Will be transformed to be TensorFlow
        compatible.

    use_cluster: str
        the column name of the clusters in the obs of the specified modality.

    """

    with_respect_to: List[str] = Field(
        default_factory=list,
        description="List of component names to compute integrated gradients with respect to their latent representations. Will be transformed to be TensorFlow compatible.",
    )
    use_cluster: str = Field(
        description="The column name of the clusters in the obs of the specified modality.",
    )

    @field_validator("with_respect_to", mode="after")
    def convert_wrt_to_tensorflow_compatible_list_of_string(cls, value: List[str]):
        return [GeneralUtils.convert_to_tensorflow_compatible_string(v) for v in value]
