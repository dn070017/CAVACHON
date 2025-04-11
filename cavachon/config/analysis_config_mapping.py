from typing import Any, List, Mapping

from cavachon.config.config_mapping.analysis_attribution_score_config_mapping import (
    AnalysisAttributionScoreConfigMapping,
)
from cavachon.config.config_mapping.analysis_generic_config_mapping import (
    AnalysisGenericConfigMapping,
)
from cavachon.config.config_mapping.analysis_visualize_embedding import (
    AnalysisVisualizeEmbedding,
)
from cavachon.config.config_mapping.config_mapping import ConfigMapping


class AnalysisConfigMapping(ConfigMapping):
    """AnalysisConfigMapping

    Config mapping for analysis.

    Attributes
    ----------
    clustering: List[AnalysisGenericConfigMapping]
        config for clustering.

    visualize_embedding: List[AnalysisVisualizeEmbeddingConfigMapping]
        config for embedding visualization.

    differential_analysis: List[AnalysisGenericConfigMapping]
        config for differential analysis.

    annotation_colnames: List[str]
        column names for the annotated cluster that needs to be
        included in the clustering analysis.

    conditional_attribution_scores: List[AnalysisAttributionScoreConfigMapping]
        config for the conditional attribution score analysis.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for AnalysisConfigMapping.

        Parameters
        ----------
        clustering: Dict[str, str], optional
            config for clustering. The keys are the modality names, the
            values are the component that is used to identify the
            clusters of modalities. Defaults to dict().

        annotation_colnames: List[str], optional
            column names for the annotated cluster that needs to be
            included in the clustering analysis. Needs to match the
            column names in adata.obs. Defaults to [].

        conditional_attribution_scores: List[AnalysisAttributionScoreConfigMapping], optional
            config for the conditional attribution score analysis.
            Defaults to [].

        """
        # change default values here
        self.clustering: Mapping[str, str] = dict()
        self.differential_analysis: Mapping[str, str] = dict()
        self.visualize_embedding: Mapping[str, str] = dict()
        self.annotation_colnames: List[str] = []
        self.conditional_attribution_scores: List[
            AnalysisAttributionScoreConfigMapping
        ] = []

        super().__init__(
            kwargs,
            [
                "clustering",
                "differential_analysis",
                "visualize_embedding",
                "annotation_colnames",
                "conditional_attribution_scores",
            ],
        )

        self.clustering = [AnalysisGenericConfigMapping(**x) for x in self.clustering]

        self.visualize_embedding = [
            AnalysisVisualizeEmbedding(**x) for x in self.visualize_embedding
        ]

        self.differential_analysis = [
            AnalysisGenericConfigMapping(**x) for x in self.differential_analysis
        ]

        self.conditional_attribution_scores = [
            AnalysisAttributionScoreConfigMapping(**x)
            for x in self.conditional_attribution_scores
        ]
