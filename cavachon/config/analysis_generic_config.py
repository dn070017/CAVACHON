from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.utils.general_utils import GeneralUtils


class AnalysisGenericConfigMapping(ConfigMapping):
    """AnalysisGenericConfigMapping

    Config mapping for analysis which uses modality and component.

    Attributes
    ----------
    modality: str
        which modality of the outputs of the component to used.

    component: str
        the outputs of which component to used.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for AnalysisGenericConfigMapping.

        Parameters
        ----------
        modality: str
            which modality of the outputs of the component to used.

        component: str
            the outputs of which component to used.

        """
        # change default values here
        self.modality: str = ""
        self.component: str = ""

        super().__init__(kwargs)
        self.modality = GeneralUtils.tensorflow_compatible_str(self.modality)
        self.component = GeneralUtils.tensorflow_compatible_str(self.component)
