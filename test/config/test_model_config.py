import pytest
from pydantic import ValidationError

from cavachon.config.model_component_config import ModelComponentConfig
from cavachon.config.model_config import ModelConfig


@pytest.fixture
def sample_component_config_data():
    return {"name": "TestComponent", "modalities": ["rna"]}


@pytest.fixture
def sample_component_config(sample_component_config_data):
    return ModelComponentConfig(**sample_component_config_data)


def test_model_config_defaults(sample_component_config_data):
    config_data = {"components": [sample_component_config_data]}
    config = ModelConfig(**config_data)

    assert config.name == "cavachon"
    assert config.load_weights is False
    assert len(config.components) == 1
    assert isinstance(config.components[0], ModelComponentConfig)
    assert config.components[0].name == "testcomponent"


def test_model_config_custom_values():
    component_data_1 = {"name": "Component_1", "modalities": ["rna"]}
    component_data_2 = {"name": "Component 2", "modalities": ["atac"]}
    config_data = {
        "name": "MyCustomModel",
        "components": [component_data_1, component_data_2],
        "load_weights": True,
    }
    config = ModelConfig(**config_data)

    assert config.name == "mycustommodel"
    assert config.load_weights is True
    assert len(config.components) == 2
    assert isinstance(config.components[0], ModelComponentConfig)
    assert isinstance(config.components[1], ModelComponentConfig)
    assert config.components[0].name == "component_1"
    assert config.components[1].name == "component_2"


def test_model_config_name_validation(sample_component_config):
    config = ModelConfig(
        name="Model Name With Spaces", components=[sample_component_config]
    )
    assert config.name == "model_name_with_spaces"

    # Test assignment
    config.name = "Another_Name-123"
    assert config.name == "another_name-123"  # Validator converts to lowercase


def test_model_config_missing_components():
    config_data = {
        "name": "TestModel",
        "load_weights": False,
        # components is missing
    }
    with pytest.raises(ValidationError) as excinfo:
        ModelConfig(**config_data)
    assert "components" in str(excinfo.value)


def test_model_config_invalid_types(sample_component_config_data):
    base_data = {"components": [sample_component_config_data]}

    # Invalid name type (should be str or None)
    # Pydantic v2 might handle int conversion, let's test with a clearly invalid type like list
    with pytest.raises(ValidationError):
        ModelConfig(**base_data, name=[])

    # Invalid components type (not a list)
    with pytest.raises(ValidationError):
        ModelConfig(name="Test", components="not_a_list")

    # Invalid item type in components list
    with pytest.raises(ValidationError):
        ModelConfig(name="Test", components=["not_a_component_config"])

    # Invalid load_weights type (not a bool)
    with pytest.raises(ValidationError):
        ModelConfig(**base_data, load_weights="not_a_bool")


def test_model_config_assignment(sample_component_config):
    config = ModelConfig(components=[sample_component_config])

    assert config.name == "cavachon"
    assert config.load_weights is False

    config.name = "NewModelName"
    assert config.name == "newmodelname"

    config.load_weights = True
    assert config.load_weights is True

    new_component = ModelComponentConfig(name="NewComponent", modalities=["atac"])
    config.components = [sample_component_config, new_component]
    assert len(config.components) == 2
    assert config.components[1].name == "newcomponent"

    # Test invalid assignment
    with pytest.raises(ValidationError):
        config.name = 123

    with pytest.raises(ValidationError):
        config.load_weights = "invalid"

    with pytest.raises(ValidationError):
        config.components = "not_a_list"

    with pytest.raises(ValidationError):
        config.components = [1, 2, 3]
