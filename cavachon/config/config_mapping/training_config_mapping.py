from typing import Any, Mapping

from cavachon.config.config_mapping.config_mapping import ConfigMapping
from cavachon.config.config_mapping.optimizer_config_mapping import (
    OptimizerConfigMapping,
)


class TrainingConfigMapping(ConfigMapping):
    """TrainingConfigMapping

    Config mapping for training.

    Attributes
    ----------
    optimizer: OptimizerConfigMapping
        config for optimizer.

    max_n_epochs: int
        maximum number of epochs for training.

    train: bool
        whether or not to retrain (finetune) the model.

    early_stopping: bool
        whether or not to use early stopping when training the model.
        Ignored if `train=False`.
    """

    def __init__(self, **kwargs: Mapping[str, Any]):
        """Constructor for TrainingConfigMapping.

        Parameters
        ----------
        optimizer: OptimizerConfigMapping
            optimizer config.

        max_n_epochs: int, optional
            maximum number of epochs for training. Defaults to 500.

        train: bool, optional
            whether or not to retrain (finetune) the model. Defaults to
            True.

        early_stopping: bool, optional
            whether or not to use early stopping when training the
            model. Ignored if `train=False`. Defaults to True.
        """
        # change default values here
        self.optimizer: OptimizerConfigMapping
        self.max_n_epochs: int = 500
        self.train: bool = True
        self.early_stopping: bool = True
        super().__init__(
            kwargs, ["optimizer", "max_n_epochs", "train", "early_stopping"]
        )

        # postprocessing
        self.setdefault("optimizer", {"name": "adam", "learning_rate": 1e-4})
        self.optimizer = OptimizerConfigMapping(**self.optimizer)
