from typing import List

from pydantic import BaseModel, ConfigDict, Field

from cavachon.config.analysis_attribution_score_config import (
    AnalysisAttributionScoreConfig,
)
from cavachon.config.analysis_generic_config import AnalysisGenericConfig
from cavachon.config.analysis_visualize_embedding_config import (
    AnalysisVisualizeEmbeddingConfig,
)


class AnalysisConfig(BaseModel):
    """AnalysisConfig

    Config for various analysis tasks.

    Attributes
    ----------
    clustering: List[AnalysisGenericConfig]
        list of configurations for clustering analysis. Each item specifies
        a modality and the component representation to use for clustering.

    visualize_embedding: List[AnalysisVisualizeEmbeddingConfig]
        list of configurations for embedding visualization. Each item defines
        the modality, representation, embedding method, coloring scheme,
        and interactivity.

    differential_analysis: List[AnalysisGenericConfig]
        list of configurations for differential analysis. Each item specifies
        a modality and the component representation to use.

    conditional_attribution_scores: List[AnalysisAttributionScoreConfig]
        list of configurations for conditional attribution score analysis.
        Each item specifies the target modality/component, the components
        to compute gradients with respect to, and the clustering column to use.
    """

    clustering: List[AnalysisGenericConfig] = Field(
        default_factory=list,
        description="list of configurations for clustering analysis.",
    )
    visualize_embedding: List[AnalysisVisualizeEmbeddingConfig] = Field(
        default_factory=list,
        description="list of configurations for embedding visualization.",
    )
    differential_analysis: List[AnalysisGenericConfig] = Field(
        default_factory=list,
        description="list of configurations for differential analysis.",
    )
    conditional_attribution_scores: List[AnalysisAttributionScoreConfig] = Field(
        default_factory=list,
        description="list of configurations for conditional attribution score analysis.",
    )

    model_config = ConfigDict(
        extra="forbid", revalidate_instances="always", validate_assignment=True
    )
