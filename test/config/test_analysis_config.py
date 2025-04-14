from contextlib import nullcontext

import pytest
from pydantic import ValidationError

from cavachon.config.analysis_attribution_score_config import (
    AnalysisAttributionScoreConfig,
)
from cavachon.config.analysis_config import AnalysisConfig
from cavachon.config.analysis_generic_config import AnalysisGenericConfig
from cavachon.config.analysis_visualize_embedding_config import (
    AnalysisVisualizeEmbeddingConfig,
)


def test_analysis_config_defaults():
    config = AnalysisConfig()

    # Check fields and types
    assert hasattr(config, "clustering")
    assert isinstance(config.clustering, list)
    assert len(config.clustering) == 0

    assert hasattr(config, "visualize_embedding")
    assert isinstance(config.visualize_embedding, list)
    assert len(config.visualize_embedding) == 0

    assert hasattr(config, "differential_analysis")
    assert isinstance(config.differential_analysis, list)
    assert len(config.differential_analysis) == 0

    assert hasattr(config, "conditional_attribution_scores")
    assert isinstance(config.conditional_attribution_scores, list)
    assert len(config.conditional_attribution_scores) == 0


def test_analysis_config_instantiation_valid():
    valid_data = {
        "clustering": [AnalysisGenericConfig(modality="test", component="test")],
        "visualize_embedding": [
            AnalysisVisualizeEmbeddingConfig(
                modality="test", use_rep="use_rep", color_by="test"
            )
        ],
        "differential_analysis": [
            AnalysisGenericConfig(modality="test", component="test")
        ],
        "conditional_attribution_scores": [
            AnalysisAttributionScoreConfig(
                modality="test",
                component="test",
                use_cluster="test",
                with_respect_to=["test"],
            )
        ],
    }

    with nullcontext():
        AnalysisConfig(**valid_data)


def test_initialization_with_invalid_types():
    invalid_data = {
        "clustering": "invalid",
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "visualize_embedding": "invalid",
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "visualize_embedding": "invalid",
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "conditional_attribution_scores": "invalid",
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "clustering": ["invalid"],
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "visualize_embedding": ["invalid"],
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "visualize_embedding": ["invalid"],
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

    invalid_data = {
        "conditional_attribution_scores": ["invalid"],
    }
    with pytest.raises(ValidationError):
        AnalysisConfig(**invalid_data)

def test_assignment_with_invalid_types():
    config = AnalysisConfig()
    with pytest.raises(ValidationError):
        config.clustering = "invalid"

    with pytest.raises(ValidationError):
        config.visualize_embedding = "invalid"

    with pytest.raises(ValidationError):
        config.differential_analysis = "invalid"

    with pytest.raises(ValidationError):
        config.conditional_attribution_scores = "invalid"

    with pytest.raises(ValidationError):
        config.clustering = ["invalid"]

    with pytest.raises(ValidationError):
        config.visualize_embedding = ["invalid"]

    with pytest.raises(ValidationError):
        config.differential_analysis = ["invalid"]

    with pytest.raises(ValidationError):
        config.conditional_attribution_scores = ["invalid"]