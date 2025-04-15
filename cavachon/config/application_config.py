import os
from typing import List, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cavachon.config.analysis_config import AnalysisConfig
from cavachon.config.dataset_config import DatasetConfig
from cavachon.config.io_config import IOConfig
from cavachon.config.model_component_config import ModelComponentConfig
from cavachon.config.model_config import ModelConfig
from cavachon.config.training_config import TrainingConfig


class ApplicationConfig(BaseModel):
    """ApplicationConfig

    Data structure for the configuration for CAVACHON application.

    Attributes
    ----------
    io: IOConfig
        IO related config.

    model: ModelConfig
        model related config.

    analysis: AnalysisConfig
        analysis related config.

    training: TrainingConfig
        training related config.

    dataset: DatasetConfig = Field(exclude=True)
        dataset related config.

    """

    io: IOConfig | None = Field(
        default_factory=IOConfig, description="io related config."
    )
    model: ModelConfig = Field(description="model related config.")
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="analysis related config."
    )
    training: TrainingConfig | None = Field(
        default_factory=TrainingConfig, description="training related config."
    )
    dataset: DatasetConfig = Field(description="dataset related config.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
        populate_by_name=True,
    )

    @staticmethod
    def check_list_of_base_model_contains_name(
        base_models: List[BaseModel], name: str
    ) -> bool:
        """Check if a list of Pydantic BaseModels contains an instance
        with a specific name.

        Parameters
        ----------
        base_models: List[BaseModel]
            list of Pydantic BaseModel instances. Each instance must
            have a 'name' attribute.

        name: str
            the name to search for.

        Returns
        -------
        bool
            True if an instance with the specified name is found,
            False otherwise.
        """
        for base_model in base_models:
            if base_model.name == name:
                return True
        return False

    def check_parent_component_exists(
        self,
        target_component_config: ModelComponentConfig,
    ) -> None:
        """Validate if parent components specified in a component
        config exist.

        Checks both `conditioned_on_z` and `conditioned_on_z_hat`
        attributes of the `target_component_config` to ensure the
        specified parent component names exist within the
        `self.model.components` list.

        Parameters
        ----------
        target_component_config : ModelComponentConfig
            the configuration of the component whose parent
            dependencies are being checked.

        Raises
        ------
        ValueError
            if a specified parent component name is not found in
            `self.model.components`.
        """
        for parent_component_name in target_component_config.conditioned_on_z:
            if not ApplicationConfig.check_list_of_base_model_contains_name(
                base_models=self.model.components, name=parent_component_name
            ):
                raise ValueError(
                    f"parent component {parent_component_name} of component {target_component_config.name} is not in the model.components."
                )
        for parent_component_name in target_component_config.conditioned_on_z_hat:
            if not ApplicationConfig.check_list_of_base_model_contains_name(
                base_models=self.model.components, name=parent_component_name
            ):
                raise ValueError(
                    f"parent component {parent_component_name} of component {target_component_config.name} is not in the model.components."
                )

    def check_modality_exists(
        self,
        target_component_config: ModelComponentConfig,
    ) -> None:
        """Validate if modalities specified in a component config exist
        in the dataset config.

        Checks the `modalities` attribute of the
        `target_component_config` to ensure the specified modality
        names exist within the `self.dataset.modalities` list.

        Parameters
        ----------
        target_component_config : ModelComponentConfig
            the configuration of the component whose modality
            dependencies are being checked.

        Raises
        ------
        ValueError
            if a specified modality name is not found in
            `self.dataset.modalities`.
        """
        for modality_name in target_component_config.modalities:
            if not ApplicationConfig.check_list_of_base_model_contains_name(
                self.dataset.modalities, modality_name
            ):
                raise ValueError(
                    f"modality {modality_name} of component {target_component_config.name} is not in the dataset.modalities."
                )

    def validate_component_configs(self) -> None:
        """Validate all component configurations within the model config.

        Iterates through each component configuration in
        `self.model.components` and performs validation checks for
        parent component existence and modality existence using
        `check_parent_component_exists` and `check_modality_exists`.
        """
        for component_config in self.model.components:
            self.check_parent_component_exists(
                target_component_config=component_config,
            )
            self.check_modality_exists(
                target_component_config=component_config,
            )

    def check_modality_and_component(
        self,
        modality_name: str,
        component_name: str,
    ) -> None:
        """Validate if a specific component exists and contains a
        specific modality.

        Checks if a component with `component_name` exists in
        `self.model.components`. If it exists, it further checks if
        `modality_name` is listed within that component's `modalities`
        attribute.

        Parameters
        ----------
        modality_name : str
            the name of the modality to check for within the component.

        component_name : str
            the name of the component to check.

        Raises
        ------
        ValueError
            if the following occurs:
            1. the specified component is not found in
                `self.model.components`.
            2. the specified modality is not found within the specified
                component's`modalities` list.
        """
        has_component = False
        for component_config in self.model.components:
            if component_config.name == component_name:
                if modality_name not in component_config.modalities:
                    raise ValueError(
                        f"modality {modality_name} of component {component_name} is not in the dataset.modalities."
                    )
                has_component = True

        if not has_component:
            raise ValueError(
                f"component {component_name} is not in the model.components."
            )

    def check_modality(
        self,
        modality_name: str,
    ) -> None:
        """Validate if a specific modality exists in the dataset config.

        Checks if a modality with `modality_name` exists within the
        `self.dataset.modalities` list.

        Parameters
        ----------
        modality_name : str
            the name of the modality to check.

        Raises
        ------
        ValueError
            if the specified modality name is not found in
            `self.dataset.modalities`.
        """
        if not ApplicationConfig.check_list_of_base_model_contains_name(
            self.dataset.modalities, modality_name
        ):
            raise ValueError(
                f"modality {modality_name} is not in the dataset.modalities."
            )

    def validate_analysis_configs(self) -> None:
        """Validate all analysis configurations.

        Iterates through configurations for conditional attribution
        scores, clustering, differential analysis, and embedding
        visualization, performing checks to ensure specified components
        and modalities exist using `check_modality_and_component` and
        `check_modality`.
        """
        for attribution_score_config in self.analysis.conditional_attribution_scores:
            self.check_modality_and_component(
                modality_name=attribution_score_config.modality,
                component_name=attribution_score_config.component,
            )

        for clustering_config in self.analysis.clustering:
            self.check_modality_and_component(
                modality_name=clustering_config.modality,
                component_name=clustering_config.component,
            )

        for differential_analysis_config in self.analysis.differential_analysis:
            self.check_modality_and_component(
                modality_name=differential_analysis_config.modality,
                component_name=differential_analysis_config.component,
            )

        for visualize_embedding_config in self.analysis.visualize_embedding:
            self.check_modality(modality_name=visualize_embedding_config.modality)

    @model_validator(mode="after")
    def validate_model(cls, instance: Self) -> Self:
        """Pydantic model validator executed after model initialization.

        Calls `validate_component_configs` and
        `validate_analysis_configs` to perform comprehensive validation
        across component and analysis settings.

        Parameters
        ----------
        instance : Self
            The ApplicationConfig instance being validated.

        Returns
        -------
        ApplicationConfig
            The validated ApplicationConfig instance.
        """
        instance.validate_component_configs()
        instance.validate_analysis_configs()

        return instance

    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        """Loads configuration from a YAML file.

        Parameters
        ----------
        filename: str
            path to the yaml config file.

        Returns
        -------
        ApplicationConfig
            The validated ApplicationConfig instance.
        """
        realpath = os.path.realpath(filename)
        with open(realpath, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            if yaml_config is None:
                raise ValueError(f"YAML file {realpath} is empty.")
        return cls(**yaml_config)
        return cls(**yaml_config)
