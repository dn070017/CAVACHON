import os
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List, Mapping

import yaml

from cavachon.config.config_mapping.analysis_config_mapping import AnalysisConfigMapping
from cavachon.config.config_mapping.component_config_mapping import (
    ComponentConfigMapping,
)
from cavachon.config.config_mapping.dataset_config_mapping import DatasetConfigMapping
from cavachon.config.config_mapping.filter_config_mapping import FilterConfigMapping
from cavachon.config.config_mapping.io_config_mapping import IOConfigMapping
from cavachon.config.config_mapping.modality_config_mapping import ModalityConfigMapping
from cavachon.config.config_mapping.model_config_mapping import ModelConfigMapping
from cavachon.config.config_mapping.sample_config_mapping import SampleConfigMapping
from cavachon.config.config_mapping.training_config_mapping import TrainingConfigMapping
from cavachon.environment.constants import Constants
from cavachon.utils.general_utils import GeneralUtils


class ApplicationConfig:
    """ApplicationConfig

    Data structure for the configuration for CAVACHON application.

    Attributes
    ----------
    filename: str
        filename of the config in YAML format.

    analysis: AnalysisConfigMapping
        Analysis related config.

    io: IOConfigMapping
        IO related config.

    sample: OrderedDict[str, SampleConfigMapping]
        sample related config, where the key is the sample name, the
        value is the corresponding SampleConfigMapping.

    modality: Dict[str, ModalityConfigMapping]
        modality related config, where the key is the modality name,
        the value is the corresponding ModalityConfigMapping.

    modality_names: List[str]
        all used modality names.

    filter: Dict[str, List[FilterConfigMapping]]
        modality filter steps related config, where the key is the name
        of the modality to filter, the value is a list of config for
        filtering steps.

    model: ModelConfigMapping
        model related config.

    training: TrainingConfigMapping
        training related config.

    dataset: DatasetConfigMapping
        dataset related config.

    components: List[ComponentConfigMapping]
        the topological sorted (based on dependency graph) list of
        components related config.

    yaml: Dict[str, Any]
        the original yaml config in dictionary format.

    """

    def __init__(self, filename: str) -> None:
        """Constructor for Config instance.

        Parameters
        ----------
        filenames: str
            filename of the config in YAML format.

        Raises
        ------
        KeyError
            if any of the required key is not in the provided config.

        See Also
        --------
        setup_io: setup the io config and datadir.
        setup_analysis: setup analysis related config.
        setup_modality: setup modality related config.
        setup_sample: setup sample related config.
        setup_training: setup training related config.
        setup_dataset: setup dataset related config.
        setup_model: setup model related config.

        """
        self.filename = os.path.realpath(filename)
        with open(filename, "r") as f:
            self.yaml: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

        # initializations
        self.analysis: AnalysisConfigMapping = None
        self.io: IOConfigMapping = None
        self.sample: OrderedDict[str, SampleConfigMapping] = OrderedDict()
        self.modality: Dict[str, ModalityConfigMapping] = dict()
        self.modality_names: List[str] = list()
        self.model: ModelConfigMapping = None
        self.filter: Dict[str, List[FilterConfigMapping]] = dict()
        self.training: TrainingConfigMapping = None
        self.components: List[ComponentConfigMapping] = list()
        self.dataset: DatasetConfigMapping = None

        # set defaults values, preprocessing the configs
        self.setup_io()
        self.setup_modality()
        self.setup_sample()
        self.setup_training()
        self.setup_dataset()
        self.setup_model()
        self.setup_analysis()

        return

    def are_all_fields_in_mapping(
        self, key_list: List[Any], mapping: Mapping, field: str, subfield: str = ""
    ) -> bool:
        """Check if all the required keys are in the provided mapping.

        Parameters
        ----------
        key_list: List[Any]:
            the required list of keys.

        mapping: Mapping
            the mapping to be evaluated.

        field: str
            the field of config (only used for error message)

        subfield: str, optional
            the subfield of config (only used for error message).
            Defaults to ''.

        Returns
        -------
        bool
            whether all the required keys are in the provided mapping.

        Raises
        ------
        KeyError
            if any of the required key is not in the provided config.

        """
        keys_not_exist = []
        for key in key_list:
            if key not in mapping:
                keys_not_exist.append(key)

        all_required_keys_are_there = len(keys_not_exist) == 0
        if not all_required_keys_are_there:
            message = ""
            for key in keys_not_exist:
                if subfield != "":
                    subfield = f" in the {subfield}"
                message += "".join(
                    (
                        f"{key} is required{subfield} for ",
                        f"{field} in the config file {self.filename}.\n",
                    )
                )
            raise KeyError(message)

        return all_required_keys_are_there

    def setup_io(self) -> None:
        """Setup IO related config."""
        self.io = IOConfigMapping(**self.yaml.get(Constants.CONFIG_FIELD_IO))

    def setup_modality(self) -> None:
        """Setup modality related config and modality names. This
        function does the following:
        1.  Setup config for modality, transform the list of modality
            config to a dictionary where the keys are the modality
            names, and the values are the configuration for the
            modalities.
        2.  Check all the relevant fields are in the config. Set
            defaults if the optional fields are not provided.

        Raises
        ------
        KeyError
            if any of the required key is not in the provided config.

        """
        # Clear self.modality and self.filter
        self.modality = dict()
        self.filter = dict()

        # Get the list of modality config
        modality_config_list = self.yaml.get(Constants.CONFIG_FIELD_MODALITY, [])
        if len(modality_config_list) == 0:
            raise KeyError(f"No modality found in the config file {self.filename}.")

        for i, modality_config in enumerate(modality_config_list):
            # Check if all required keys are in the modality config.
            self.are_all_fields_in_mapping(
                Constants.CONFIG_FIELD_MODALITY_REQUIRED,
                modality_config,
                Constants.CONFIG_FIELD_MODALITY,
            )

            # Create the ModalityConfigMapping instance
            modality_config = ModalityConfigMapping(**modality_config)

            # Save the processed result to self.modality and self.filter
            modality_name = modality_config.name
            filters_config = modality_config.filters
            self.modality.setdefault(modality_name, modality_config)
            self.filter.setdefault(modality_name, filters_config)

        # Save the modality names
        self.modality_names = list(self.modality.keys())

        return

    def setup_sample(self) -> None:
        """Setup sample related config. This function does the
        following:
        1.  Setup config for sample, transform the list of sample
            config to a OrderedDict where the keys are the sample names,
            and the values are the configuration for the sample. The
            order of the insertion depends on the order the samples
            appear.
        2.  Save the sample modality config to each modality in the
            config_modality.
        3.  Check all the relevant fields are in the config. Set
            defaults if the optional fields are not provided.

        Raises
        ------
        KeyError
            if any of the required key is not in the provided config.

        """
        # Clear the OrderedDict of config sample
        self.sample = OrderedDict()
        # Get the list of sample config
        sample_config_list = self.yaml.get(Constants.CONFIG_FIELD_SAMPLE, [])
        for i, sample_config in enumerate(sample_config_list):
            # Check if all required keys are in the sample config.
            sample_name = sample_config.get("name")
            self.are_all_fields_in_mapping(
                Constants.CONFIG_FIELD_SAMPLE_REQUIRED,
                sample_config,
                Constants.CONFIG_FIELD_SAMPLE,
            )

            # Check each modality config in the sample
            for i, sample_modality_config in enumerate(
                sample_config.get(Constants.CONFIG_FIELD_MODALITY)
            ):
                # Check if all required keys are in the modality of the sample.
                self.are_all_fields_in_mapping(
                    Constants.CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED,
                    sample_modality_config,
                    sample_name,
                    Constants.CONFIG_FIELD_MODALITY,
                )

                # Check if the modality name is in the self.modality
                modality_name = sample_modality_config.get("name")
                modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
                if modality_name in self.modality:
                    # Put the sample names into self.modality[`modality_name`][`sample`]
                    self.modality.get(modality_name).get(
                        Constants.CONFIG_FIELD_SAMPLE
                    ).append(sample_name)
                else:
                    message = "".join(
                        (
                            f"No configuration for modality {modality_name} of sample {sample_name} ",
                            f"in the config file {self.filename}.",
                        )
                    )
                    raise KeyError(message)

        for i, sample_config in enumerate(sample_config_list):
            self.sample.setdefault(sample_name, SampleConfigMapping(**sample_config))

        return

    def setup_training(self) -> None:
        """Setup training related config."""
        model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
        training_config = model_config.get(Constants.CONFIG_FIELD_MODEL_TRAINING, {})
        self.training = TrainingConfigMapping(**training_config)

    def setup_dataset(self) -> None:
        """Setup dataset related config."""
        model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
        dataset_config = model_config.get(Constants.CONFIG_FIELD_MODEL_DATASET, {})
        self.dataset = DatasetConfigMapping(**dataset_config)

    def setup_model(self) -> None:
        """Setup model and component related config.

        Raises
        ------
        KeyError
            if any of the required key is not in the provided config.

        AttributeError
            if the dependencies between components is not a directed
            acyclic graph.
        """
        self.model = None
        self.components = list()

        model_config = self.yaml.get(Constants.CONFIG_FIELD_MODEL, {})
        self.are_all_fields_in_mapping(
            Constants.CONFIG_FIELD_MODEL_REQUIRED, model_config, "model config"
        )

        # Check required field and default values if not specified in the component config
        component_config_list = model_config.get(Constants.CONFIG_FIELD_MODEL_COMPONENT)
        component_config_mapping = dict()
        for i, component_config in enumerate(component_config_list):
            # Check required field
            self.are_all_fields_in_mapping(
                Constants.CONFIG_FIELD_COMPONENT_REQUIRED, component_config, "component"
            )
            component_name = component_config.get("name")

            # Setup modality names and decoder for the component
            for j, modality_config in enumerate(
                component_config.get(Constants.CONFIG_FIELD_MODALITY)
            ):
                self.are_all_fields_in_mapping(
                    Constants.CONFIG_FIELD_COMPONENT_MODALITIES_REQUIRED,
                    modality_config,
                    component_name,
                    "modalities config",
                )
                modality_name = modality_config.get("name")
                modality_name = GeneralUtils.tensorflow_compatible_str(modality_name)
                component_config[Constants.CONFIG_FIELD_MODALITY][j]["name"] = (
                    modality_name
                )
                if modality_name not in self.modality:
                    message = f"'{modality_name}' is not in the config of modality."
                    raise KeyError(message)
                modality_dist = self.modality.get(modality_name).get(
                    Constants.CONFIG_FIELD_MODALITY_DIST
                )
                modality_config[
                    Constants.CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES
                ] = modality_dist

            component_config = ComponentConfigMapping(**component_config)
            component_config_mapping.setdefault(component_name, component_config)

        # Sort the components based on the BFS order
        self.components = GeneralUtils.order_components(component_config_mapping)

        model_fields = [
            "name",
            Constants.CONFIG_FIELD_MODEL_LOAD_WEIGHTS,
            Constants.CONFIG_FIELD_MODEL_SAVE_WEIGHTS,
            Constants.CONFIG_FIELD_MODEL_COMPONENT,
            Constants.CONFIG_FIELD_MODEL_TRAINING,
            Constants.CONFIG_FIELD_MODEL_DATASET,
        ]

        self.model = ModelConfigMapping(
            **{field: model_config.get(field) for field in model_fields}
        )

    def setup_analysis(self) -> None:
        """Setup analysis related config.

        Raises
        ------
        KeyError
            if
            1. `modality` is not in the config of component `component`.
            2. `component` is not in the config of components.
            3. `with_respect_to` is not in the config of components.
        """
        self.analysis = AnalysisConfigMapping(
            **self.yaml.get(Constants.CONFIG_FIELD_ANALYSIS)
        )

        for specific_analysis_config in chain(
            self.analysis.clustering,
            self.analysis.differential_analysis,
            self.analysis.conditional_attribution_scores,
        ):
            modality = specific_analysis_config.modality
            component = specific_analysis_config.component
            has_component = False
            has_modality = False
            for component_config in self.components:
                if component == component_config.name:
                    has_component = True
                    for modality_in_component in component_config.modality_names:
                        if modality == modality_in_component:
                            has_modality = True
                            break
                if has_modality and has_component:
                    break
            if not has_modality:
                message = (
                    f"'{modality}' is not in the config of component '{component}'."
                )
                raise KeyError(message)
            if not has_component:
                message = f"'{component}' is not in the config of components."
                raise KeyError(message)

        for visualize_embedding_config in self.analysis.visualize_embedding:
            modality = visualize_embedding_config.modality
            has_modality = False
            for component_config in self.components:
                for modality_in_component in component_config.modality_names:
                    if modality == modality_in_component:
                        has_modality = True
                        break
            if not has_modality:
                message = (
                    f"'{modality}' is not in the config of component '{component}'."
                )
                raise KeyError(message)

        for attribution_config in self.analysis.conditional_attribution_scores:
            modality = attribution_config.modality
            component = attribution_config.component
            with_respect_to = attribution_config.with_respect_to
            is_modality_in_component = False
            is_wrt_in_component = False
            has_modality = False
            for component_config in self.components:
                for wrt in with_respect_to:
                    if wrt == component_config.name:
                        is_wrt_in_component = True
                if component == component_config.name:
                    is_modality_in_component = True
                    for modality_in_component in component_config.modality_names:
                        if modality == modality_in_component:
                            has_modality = True
                            break
                if has_modality and is_modality_in_component and is_wrt_in_component:
                    break

            if not has_modality:
                message = (
                    f"'{modality}' is not in the config of component '{component}'."
                )
                raise KeyError(message)
            if not is_modality_in_component:
                message = (
                    f"'{component}' (modality) is not in the config of components."
                )
                raise KeyError(message)
            if not is_wrt_in_component:
                message = f"'{with_respect_to}' (with_respect_to) is not in the config of components."
                raise KeyError(message)

        for clustering_config in self.analysis.clustering:
            if clustering_config.use_rep == "z_hat":
                component = clustering_config.component
                component_config = None
                for c in self.components:
                    if c.name == component:
                        component_config = c
                        break
                if component_config is None:
                    message = f"'{component}' is not in the config of components."
                    raise KeyError(message)
                if not getattr(component_config, "reparameterize_z_hat", False):
                    message = "".join(
                        (
                            f"'{component}' in the config file {self.filename} does ",
                            "not have reparameterize_z_hat set to True, so z_hat ",
                            "clustering is not allowed.",
                        )
                    )
                    raise KeyError(message)
                if len(component_config.conditioned_on_z_hat) == 0:
                    message = "".join(
                        (
                            f"'{component}' in the config file {self.filename} has no ",
                            "parents in conditioned_on_z_hat, so z_hat clustering is ",
                            "not allowed.",
                        )
                    )
                    raise KeyError(message)
