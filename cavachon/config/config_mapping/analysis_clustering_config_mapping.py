from typing import Any, Mapping

from cavachon.config.config_mapping.analysis_generic_config_mapping import (
    AnalysisGenericConfigMapping,
)
from cavachon.utils.general_utils import GeneralUtils


class AnalysisClusteringConfigMapping(AnalysisGenericConfigMapping):
    """AnalysisClusteringConfigMapping

    Config mapping for clustering analysis. Extends AnalysisGenericConfigMapping
    with a use_rep field for choosing the latent representation.

    Attributes
    ----------
    modality: str
        which modality of the outputs of the component to use.

    component: str
        the outputs of which component to use.

    use_rep: str
        which representation to use for clustering. Must be 'z' or
        'z_hat'. Defaults to 'z'.
    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for AnalysisClusteringConfigMapping.

        Parameters
        ----------
        modality: str
            which modality of the outputs of the component to use.

        component: str
            the outputs of which component to use.

        use_rep: str, optional
            which representation to use for clustering. Must be 'z' or
            'z_hat'. Defaults to 'z'.
        """
        # change default values here
        self.use_rep: str = "z"

        super().__init__(**kwargs)
        self.use_rep = GeneralUtils.tensorflow_compatible_str(self.use_rep)
        if self.use_rep not in ("z", "z_hat"):
            raise ValueError(
                f"use_rep should be one of 'z' or 'z_hat'"
            )
