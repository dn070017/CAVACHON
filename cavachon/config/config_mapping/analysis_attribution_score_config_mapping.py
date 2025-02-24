from typing import Any, List, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.utils.GeneralUtils import GeneralUtils


class AnalysisAttributionScoreConfigMapping(ConfigMapping):
    """AnalysisAttributionScoreConfigMapping

    Config mapping for attribution score analysis.

    Attributes
    ----------
    modality: str
        which modality of the outputs of the component to used.

    component: str
        the outputs of which component to used.

    with_respect_to: str
        compute integrated gradietn with respect to the latent
        representation of which component.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for AnalysisAttributionScoreConfigMapping.

        Parameters
        ----------
        modality: str
            which modality of the outputs of the component to used.

        component: str
            the outputs of which component to used.

        with_respect_to: List[str]
            compute integrated gradient with respect to the latent
            representation of which component.

        use_cluster: str
            the column name of the clusters in the obs of modality.

        """
        # change default values here
        self.modality: str = ""
        self.component: str = ""
        self.with_respect_to: List[str] = list()
        self.use_cluster: str = ""

        super().__init__(kwargs)
        self.modality = GeneralUtils.tensorflow_compatible_str(self.modality)
        self.component = GeneralUtils.tensorflow_compatible_str(self.component)
        for i, wrt in enumerate(self.with_respect_to):
            self.with_respect_to[i] = GeneralUtils.tensorflow_compatible_str(wrt)
