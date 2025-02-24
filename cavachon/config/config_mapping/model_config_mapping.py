from typing import Any, List, Mapping

from cavachon.config.config_mapping.component_config_mapping import (
    ComponentConfigMapping,
)
from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.config.config_mapping.dataset_config_mapping import DatasetConfigMapping
from cavachon.config.config_mapping.training_config_mapping import TrainingConfigMapping
from cavachon.utils.GeneralUtils import GeneralUtils


class ModelConfigMapping(ConfigMapping):
    """ModelConfigMapping

    Config mapping for model.

    Attributes
    ----------
    name: str
        name of the model.

    components: List[ComponentConfigMapping]
        list of component configs.

    training: TrainingConfigMapping
        training config.

    dataset: DatasetConfigMapping
        dataset config

    load_weights: bool
        whether or not to load the pretrained weights before training.

    save_weights: bool
        whether or not to save the weights after training.
    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for ModelConfigMapping.

        Parameters
        ----------
        name: str
            name of the model.

        components: List[ComponentConfigMapping]
            list of component configs.

        training: Union[Mapping[str, Any], TrainingConfigMapping]
            training config.

        dataset: Union[Mapping[str, Any], DatasetConfigMapping]
            dataset config

        load_weights: bool, optional
            whether or not to load the pretrained weights before
            training. Defaults to False.

        save_weights: bool, optional
            whether or not to save the weights after training. Defaults
            to True.

        """
        # change default values here
        self.name: str = "cavachon"
        self.components: List[ComponentConfigMapping] = list()
        self.training: TrainingConfigMapping
        self.dataset: DatasetConfigMapping
        self.load_weights: bool = False
        self.save_weights: bool = True

        super().__init__(
            kwargs,
            [
                "name",
                "components",
                "training",
                "dataset",
                "load_weights",
                "save_weights",
            ],
        )

        # postprocessing
        self.name = GeneralUtils.tensorflow_compatible_str(self.name)
        if not isinstance(self.training, TrainingConfigMapping):
            self.training = TrainingConfigMapping(**self.training)
        if not isinstance(self.dataset, DatasetConfigMapping):
            self.dataset = DatasetConfigMapping(**self.dataset)
