import pytest
from pydantic import ValidationError

from cavachon.config.analysis_attribution_score_config import (
    AnalysisAttributionScoreConfig,
)
from cavachon.utils.general_utils import GeneralUtils


def test_analysis_attribution_score_config_valid():
    config_data = {
        "modality": "RNA",
        "component": "Modality",
        "with_respect_to": ["Other Modality 1", "Other Modality 2"],
        "use_cluster": "leiden_clusters",
    }
    config = AnalysisAttributionScoreConfig(**config_data)

    assert config.modality == GeneralUtils.convert_to_tensorflow_compatible_string(
        "RNA"
    )
    assert config.component == GeneralUtils.convert_to_tensorflow_compatible_string(
        "Modality"
    )
    assert config.with_respect_to == [
        GeneralUtils.convert_to_tensorflow_compatible_string("Other Modality 1"),
        GeneralUtils.convert_to_tensorflow_compatible_string("Other Modality 2"),
    ]
    assert config.use_cluster == "leiden_clusters"


def test_analysis_attribution_score_config_defaults():
    config_data = {
        "modality": "ATAC",
        "component": "Modality",
        "use_cluster": "some_clusters",
        # with_respect_to is optional, defaults to []
    }
    config = AnalysisAttributionScoreConfig(**config_data)

    assert config.modality == GeneralUtils.convert_to_tensorflow_compatible_string(
        "ATAC"
    )
    assert config.component == GeneralUtils.convert_to_tensorflow_compatible_string(
        "Modality"
    )
    assert config.with_respect_to == []
    assert config.use_cluster == "some_clusters"


def test_analysis_attribution_score_config_missing_fields():
    """Test validation errors for missing required fields."""
    # missing modality
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(component="Comp", use_cluster="cluster")

    # missing component
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(modality="Mod", use_cluster="cluster")

    # missing use_cluster
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(modality="Mod", component="Comp")

    # missing all required
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig()


def test_analysis_attribution_score_config_invalid_types():
    base_data = {"modality": "RNA", "component": "Comp", "use_cluster": "cluster"}

    # invalid type for modality
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(**{**base_data, "modality": 123})

    # invalid type for component
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(**{**base_data, "component": ["list"]})

    # invalid type for with_respect_to (not a list)
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(**{**base_data, "with_respect_to": "not_a_list"})

    # invalid type for item in with_respect_to list
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(
            **{**base_data, "with_respect_to": ["valid", 123]}
        )

    # invalid type for use_cluster
    with pytest.raises(ValidationError):
        AnalysisAttributionScoreConfig(**{**base_data, "use_cluster": None})


def test_analysis_attribution_score_config_assignment():
    config = AnalysisAttributionScoreConfig(
        modality="Initial Modality",
        component="Initial Component",
        use_cluster="initial_cluster",
    )

    assert config.modality == "initial_modality"
    assert config.component == "initial_component"
    assert config.with_respect_to == []
    assert config.use_cluster == "initial_cluster"

    # Assign new values and check transformation/validation
    config.modality = "New Modality"
    assert config.modality == "new_modality"

    config.component = "New Component"
    assert config.component == "new_component"

    config.with_respect_to = ["Comp A", "Comp B"]
    assert config.with_respect_to == ["comp_a", "comp_b"]

    config.use_cluster = "new_cluster_col"
    assert config.use_cluster == "new_cluster_col"

    # Test invalid assignment type
    with pytest.raises(ValidationError):
        config.modality = 456

    with pytest.raises(ValidationError):
        config.component = None

    with pytest.raises(ValidationError):
        config.with_respect_to = "not a list"

    with pytest.raises(ValidationError):
        config.use_cluster = 123
