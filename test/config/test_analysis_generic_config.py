import pytest
from pydantic import ValidationError

from cavachon.config.analysis_generic_config import AnalysisGenericConfig
from cavachon.utils.general_utils import GeneralUtils


def test_analysis_generic_config_valid():
    """Test valid initialization of AnalysisGenericConfig."""
    config_data = {"modality": "RNA", "component": "Modality 1"}
    config = AnalysisGenericConfig(**config_data)

    assert config.modality == GeneralUtils.convert_to_tensorflow_compatible_string(
        "RNA"
    )
    assert config.component == GeneralUtils.convert_to_tensorflow_compatible_string(
        "Modality 1"
    )


def test_analysis_generic_config_missing_fields():
    # missing component
    with pytest.raises(ValidationError):
        AnalysisGenericConfig(modality="RNA")

    # missing modality
    with pytest.raises(ValidationError):
        AnalysisGenericConfig(component="Modality 1")

    # missing both
    with pytest.raises(ValidationError):
        AnalysisGenericConfig()


def test_analysis_generic_config_invalid_types():
    # invalid type for modality
    with pytest.raises(ValidationError):
        AnalysisGenericConfig(modality=123, component="Modality")

    # invalid type for component
    with pytest.raises(ValidationError):
        AnalysisGenericConfig(modality="RNA", component=["list", "is", "invalid"])


def test_analysis_generic_config_assignment():
    config = AnalysisGenericConfig(
        modality="Initial Modality", component="Initial Component"
    )

    assert config.modality == "initial_modality"
    assert config.component == "initial_component"

    # Assign new values and check transformation
    config.modality = "New Modality"
    assert config.modality == "new_modality"

    config.component = "New Component"
    assert config.component == "new_component"

    # Test invalid assignment type
    with pytest.raises(ValidationError):
        config.modality = 456

    with pytest.raises(ValidationError):
        config.component = None
