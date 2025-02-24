from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping


class ModalityFileMatrixConfig(ConfigMapping):
    """ModalityFileMatrixConfigMapping

    Config mapping for modality matrix.

    Attributes
    ----------
    filename: str
        filename of the matrix.

    transpose: bool
        if the matrix is transposed (the matrix is transposed if vars
        as rows, obs as cols).

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for ModalityFileMatrixConfigMapping

        Parameters
        ----------
        filename: str
            filename of the matrix.

        transpose: bool, optional
            if the matrix is transposed (the matrix is transposed if
            vars as rows, obs as cols). Defaults to False.

        """
        # change default values here
        filename: str  # noqa
        transpose: bool = False  # noqa
        super().__init__(kwargs, ["filename", "transpose"])
