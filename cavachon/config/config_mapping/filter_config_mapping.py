from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping


class FilterConfigMapping(ConfigMapping):
    """FilterConfigMapping

    Config mapping for Dataset.

    Attributes
    ----------
    step: str
        filter step.

    kwargs: Mapping[str, Any]
        additional parameters for cavachon.filter.AnnDataFilter

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for FilterConfigMapping.

        Parameters
        ----------
        config: Mapping[str, Any]:
            filter config in mapping format.

        """
        self.step: str
        super().__init__(kwargs)
        self.step: str
        super().__init__(kwargs)
