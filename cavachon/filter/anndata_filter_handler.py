from collections.abc import Callable
from typing import Any, Dict, List, Mapping

from collections.abc import Callable
from typing import Dict, List

import anndata

from cavachon.filter.anndata_filter import AnnDataFilter
from cavachon.utils.reflection_handler import ReflectionHandler


class AnnDataFilterHandler(Callable):
    """AnnDataFilterHandler

    Handler for AnnDataFilter used to create a series of filter steps
    to Mapping of AnnData.

    Attributes
    ----------
    steps: Dict[str, List[AnnDataFilter]]
        filtering steps stored in dictionary. The keys are the names
        for the modality, and the values is the list the AnnDataFilters
        used to filter the corresponding AnnData.

    """

    def __init__(self, steps: Dict[str, List[AnnDataFilter]]):
        """Constructor for AnnDataFilterHandler

        Attributes
        ----------
        steps: Dict[str, List[AnnDataFilter]]
            filtering steps stored in dictionary. The keys are the
            names for the modality, and the values is the list the
            AnnDataFilters used to filter the corresponding AnnData.

        """
        self.steps: Dict[str, List[AnnDataFilter]] = steps

    @classmethod
    def from_config(cls, config):
        """Create AnnDataFilterHandler from the ApplicationConfig.

        Parameters
        ----------
        config: ApplicationConfig
            the application config used to create AnnDataFilterHandler.

        Returns
        -------
        AnnDataFilterHandler:
            AnnDataFilterHandler created from the config.filter.

        """
        steps = dict()
        for modality_name, modality_filter_steps in config.filter.items():
            step_runners = []
            for filter_step in modality_filter_steps:
                step_runner_class = ReflectionHandler.get_class_by_name(
                    filter_step.step,
                    "filter",
                )
                step_runner = step_runner_class(
                    name=filter_step.step, **filter_step.model_dump()
                )
                step_runners.append(step_runner)

            steps.setdefault(modality_name, step_runners)

        return cls(steps)

    def __call__(self, target: Mapping[str, anndata.AnnData]):
        """Perform preprocessing to all AnnDatas in the provided target.
        Note that the values in the target will be filtered.

        Parameters
        ----------
        target: Mapping[str, anndata.AnnData]
            Mapping of AnnData to preprocessed.

        Returns
        -------

        """
        for modality_name, modality in target.items():
            for step_runner in self.steps.get(modality_name):
                modality = step_runner(modality)
            target[modality_name] = modality

        return target
