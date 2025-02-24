from typing import Any, List, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.config.config_mapping.modality_config_mapping import (
    ModalityFileConfigMapping,
)


class SampleConfigMapping(ConfigMapping):
    """SampleConfigMapping

    Config mapping for sample.

    Attributes
    ----------
    name: str
        name of the sample.

    description: List[str]
        description of the samples.

    modalities: List[ModalityFileConfigMapping]
        list of modality file configs associated with the sample.

    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for SampleConfigMapping.

        Parameters
        ----------
        config: Mapping[str, Any]:
            sample config in mapping format.

        """
        # change default values here
        self.name: str
        self.description: str = ""
        self.modalities: List[ModalityFileConfigMapping] = list()

        super().__init__(kwargs, ["name", "description", "modalities"])

        # postprocessing
        for i in range(len(self.modalities)):
            self.modalities[i] = ModalityFileConfigMapping(**self.modalities[i])
