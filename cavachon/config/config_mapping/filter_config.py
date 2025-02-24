from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping


class FilterConfig(ConfigMapping):
    """FilterConfig

    Config for Dataset.

    Attributes
    ----------
    step: str
        filter step.

    kwargs: Mapping[str, Any]
        additional parameters for cavachon.filter.AnnDataFilter

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for FilterConfig.

        Parameters
        ----------
        config: Mapping[str, Any]:
            filter config in mapping format.

        """
        self.step: str
        super().__init__(kwargs)
        self.step: str
        super().__init__(kwargs)
