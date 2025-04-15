import os
from contextlib import nullcontext

import pytest
from pydantic import ValidationError

from cavachon.config.analysis_attribution_score_config import (
    AnalysisAttributionScoreConfig,
)
from cavachon.config.analysis_config import AnalysisConfig
from cavachon.config.analysis_generic_config import AnalysisGenericConfig
from cavachon.config.application_config import ApplicationConfig
from cavachon.config.dataset_config import DatasetConfig
from cavachon.config.dataset_modality_config import DatasetModalityConfig
from cavachon.config.io_config import IOConfig
from cavachon.config.model_component_config import ModelComponentConfig
from cavachon.config.model_config import ModelConfig


@pytest.fixture
def minimal_io_config():
    return IOConfig()


@pytest.fixture
def minimal_analysis_config():
    return AnalysisConfig(
        clustering=[],
        visualize_embedding=[],
        differential_analysis=[],
        conditional_attribution_scores=[],
    )


@pytest.fixture
def minimal_dataset_modality_config():
    return DatasetModalityConfig(
        name="modality_1",
        dist="MultivariateNormalDiagDistribution",
    )


@pytest.fixture
def minimal_dataset_config(minimal_dataset_modality_config):
    return DatasetConfig(modalities=[minimal_dataset_modality_config])


@pytest.fixture
def minimal_model_component_config():
    return ModelComponentConfig(
        name="component_1",
        modalities=["modality_1"],
    )


@pytest.fixture
def minimal_model_config(minimal_model_component_config):
    return ModelConfig(name="model_1", components=[minimal_model_component_config])


def test_application_config_minimal_defaults(
    minimal_model_config,
    minimal_dataset_config,
):
    config_data = {
        "model": minimal_model_config,
        "dataset": minimal_dataset_config,
    }
    with nullcontext():
        ApplicationConfig(**config_data)


def test_application_config_minimal_init(
    minimal_io_config,
    minimal_analysis_config,
    minimal_model_config,
    minimal_dataset_config,
):
    config_data = {
        "io": minimal_io_config,
        "analysis": minimal_analysis_config,
        "model": minimal_model_config,
        "dataset": minimal_dataset_config,
    }
    with nullcontext():
        ApplicationConfig(**config_data)


def test_validate_component_config_missing_parent_z(
    minimal_io_config, minimal_analysis_config, minimal_dataset_config
):
    component_1 = ModelComponentConfig(
        name="component_1", n_latent_dims=5, modalities=["modality_1"]
    )
    component_2 = ModelComponentConfig(
        name="component_2",
        n_latent_dims=5,
        modalities=["modality_1"],
        conditioned_on_z=["component_non_existent"],
    )
    model_config = ModelConfig(name="model", components=[component_1, component_2])
    config_data = {
        "io": minimal_io_config,
        "model": model_config,
        "analysis": minimal_analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValidationError):
        ApplicationConfig(**config_data)


def test_validate_component_config_missing_parent_z_hat(
    minimal_io_config, minimal_analysis_config, minimal_dataset_config
):
    component_1 = ModelComponentConfig(
        name="component_1", n_latent_dims=5, modalities=["modality_1"]
    )
    component_2 = ModelComponentConfig(
        name="component_2",
        n_latent_dims=5,
        modalities=["modality_1"],
        conditioned_on_z_hat=["component_non_existent"],
    )
    model_config = ModelConfig(name="model", components=[component_1, component_2])
    config_data = {
        "io": minimal_io_config,
        "model": model_config,
        "analysis": minimal_analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValidationError):
        ApplicationConfig(**config_data)


def test_validate_component_config_missing_modality(
    minimal_io_config,
    minimal_analysis_config,
    minimal_dataset_config,
):
    component_1 = ModelComponentConfig(
        name="component1",
        n_latent_dims=5,
        modalities=["modality_non_existent"],
    )
    model_config = ModelConfig(name="model_test", components=[component_1])
    config_data = {
        "io": minimal_io_config,
        "model": model_config,
        "analysis": minimal_analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValidationError):
        ApplicationConfig(**config_data)


def test_validate_analysis_config_missing_component(
    minimal_io_config,
    minimal_model_config,
    minimal_dataset_config,
):
    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[
            AnalysisAttributionScoreConfig(
                name="attr1",
                component="component_non_existent",
                modality="modality_1",
                use_cluster="some_cluster_col",
                with_respect_to=[],
            )
        ],
        clustering=[],
        differential_analysis=[],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValueError):
        ApplicationConfig(**config_data)

    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[],
        clustering=[
            AnalysisGenericConfig(
                component="component_non_existent",
                modality="modality_1",
            )
        ],
        differential_analysis=[],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[],
        clustering=[],
        differential_analysis=[
            AnalysisGenericConfig(
                component="component_non_existent",
                modality="modality_1",
            )
        ],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValueError):
        ApplicationConfig(**config_data)


def test_validate_analysis_config_missing_modality_in_component(
    minimal_io_config, minimal_model_config, minimal_dataset_config
):
    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[
            AnalysisAttributionScoreConfig(
                name="attr1",
                component="component_non_existent",
                modality="modality_1",
                use_cluster="some_cluster_col",
                with_respect_to=[],
            )
        ],
        clustering=[],
        differential_analysis=[],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValueError):
        ApplicationConfig(**config_data)

    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[],
        clustering=[
            AnalysisGenericConfig(
                component="component_2",
                modality="modality_1",
            )
        ],
        differential_analysis=[],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValueError):
        ApplicationConfig(**config_data)

    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[],
        clustering=[],
        differential_analysis=[
            AnalysisGenericConfig(
                component="component_2",
                modality="modality_1",
            )
        ],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with pytest.raises(ValueError):
        ApplicationConfig(**config_data)


def test_validate_analysis_config_valid(
    minimal_io_config,
    minimal_model_config,
    minimal_dataset_config,
):
    analysis_config = AnalysisConfig(
        conditional_attribution_scores=[
            AnalysisAttributionScoreConfig(
                component="component_1",
                modality="modality_1",
                use_cluster="some_cluster_col",
                with_respect_to=[],
            )
        ],
        clustering=[
            AnalysisGenericConfig(
                component="component_1",
                modality="modality_1",
            )
        ],
        differential_analysis=[
            AnalysisGenericConfig(
                component="component_1",
                modality="modality_1",
            )
        ],
        visualize_embedding=[],
    )
    config_data = {
        "io": minimal_io_config,
        "model": minimal_model_config,
        "analysis": analysis_config,
        "dataset": minimal_dataset_config,
    }
    with nullcontext():
        ApplicationConfig(**config_data)


def test_from_yaml_none(tmp_path, mocker):
    empty_file = os.path.join(tmp_path, "empty.yaml")
    with open(empty_file, "w"):
        pass

    mock_load = mocker.patch("yaml.load", return_value=None)

    with pytest.raises(ValueError):
        ApplicationConfig.from_yaml(empty_file)

    mock_load.assert_called_once()
