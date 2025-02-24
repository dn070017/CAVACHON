from typing import Any, List, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping


class ModalityFileFeatureConfig(ConfigMapping):
    """ModalityFileFeatureConfig

    Config for modality files (var and obs).

    Attributes
    ----------
    filename: str
        filename of the feature table.

    has_headers: bool
        whether or not the table have headers.

    colnames: List[str]
        column names of the DataFrame.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for ModalityFileFeatureConfig

        Parameters
        ----------
        filename: str
            filename of the feature table.

        has_headers: bool, optional
            whether or not the table have headers. Defaults to False.

        colnames: List[str], optional
            column names of the DataFrame. Defatuls to [].


        """
        # change default values here
        filename: str  # noqa
        has_headers: bool = False  # noqa
        colnames: List[str] = list()  # noqa
        super().__init__(kwargs, ["filename", "has_headers", "colnames"])
