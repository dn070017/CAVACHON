from typing import Any, List, Mapping

from cavachon.config.config_mapping.analysis_attribution_score_config_mapping import (
    AnalysisAttributionScoreConfigMapping,
)
from cavachon.config.config_mapping.analysis_generic_config_mapping import (
    AnalysisGenericConfigMapping,
)
from cavachon.config.config_mapping.config_mapping import ConfigMapping


class AnalysisConfigMapping(ConfigMapping):
    """AnalysisConfigMapping

    Config mapping for analysis.

    Attributes
    ----------
    clustering: Dict[str, str]
        config for clustering. The keys are the modality names, the
        values are the component that is used to identify the clusters
        of modalities.

    differential_analysis: Dict[str, str]
        config for differential analysis. The keys are the modality
        to be analyzed, the values are the component of which that
        generate the modality.

    embedding_methods: List[str]
        embedding methods used for downstream analysis.

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

        embedding_methods: List[str], optional
            embedding methods used for downstream analysis. Defaults to
            ['tsne'].

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
        self.embedding_methods: List[str] = ["tsne"]
        self.annotation_colnames: List[str] = []
        self.conditional_attribution_scores: List[
            AnalysisAttributionScoreConfigMapping
        ] = []

        super().__init__(
            kwargs,
            [
                "clustering",
                "differential_analysis",
                "embedding_methods",
                "annotation_colnames",
                "conditional_attribution_scores",
            ],
        )

        self.clustering = [AnalysisGenericConfigMapping(**x) for x in self.clustering]

        self.differential_analysis = [
            AnalysisGenericConfigMapping(**x) for x in self.differential_analysis
        ]

        self.conditional_attribution_scores = [
            AnalysisAttributionScoreConfigMapping(**x)
            for x in self.conditional_attribution_scores
        ]
