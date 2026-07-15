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

    min_n_obs: int
        minimum number of observations required for a cluster to be
        kept. Clusters with fewer observations are labeled as
        "Unassigned" (for z_hat) or iteratively removed (for z).
        Defaults to 36.
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

        min_n_obs: int, optional
            minimum number of observations required for a cluster to
            be kept. Defaults to 36.
        """
        # change default values here
        self.use_rep: str = "z"
        self.min_n_obs: int = 36

        super().__init__(**kwargs)
        self.use_rep = GeneralUtils.tensorflow_compatible_str(self.use_rep)
        if self.use_rep not in ("z", "z_hat"):
            raise ValueError(
                f"use_rep should be one of 'z' or 'z_hat'"
            )
        if not isinstance(self.min_n_obs, int) or self.min_n_obs <= 0:
            raise ValueError(
                f"min_n_obs should be a positive integer, got {self.min_n_obs}"
            )
