import pytest
from pydantic import ValidationError

# Corrected import path based on file location
from cavachon.config.model_component_config import ModelComponentConfig


@pytest.fixture
def modality_names():
    """Fixture for creating sample modality names list."""
    return ["rna", "atac"]


def test_component_config_defaults(modality_names):
    """Test ModelComponentConfig initialization with default values."""
    config_data = {"name": "MyComponent", "modalities": modality_names}
    config = ModelComponentConfig(**config_data)

    assert config.name == "mycomponent"
    assert config.modalities == ["rna", "atac"]
    assert isinstance(config.modalities, list)
    assert all(isinstance(m, str) for m in config.modalities)
    assert config.conditioned_on_z == []
    assert config.conditioned_on_z_hat == []
    assert config.n_latent_dims == 5
    assert config.n_latent_priors == 11
    assert config.n_encoder_layers == 3
    assert config.n_decoder_layers == 3
    assert config.n_progressive_epochs == 1


def test_component_config_custom_values(modality_names):
    config_data = {
        "name": "Custom Component",
        "modalities": modality_names,
        "conditioned_on_z": ["other_comp_1"],
        "conditioned_on_z_hat": ["other_comp_2", "other_comp_3"],
        "n_latent_dims": 10,
        "n_latent_priors": 20,
        "n_encoder_layers": 4,
        "n_decoder_layers": 2,
        "n_progressive_epochs": 5,
    }
    config = ModelComponentConfig(**config_data)

    assert config.name == "custom_component"
    assert config.modalities == ["rna", "atac"]
    assert config.conditioned_on_z == ["other_comp_1"]
    assert config.conditioned_on_z_hat == ["other_comp_2", "other_comp_3"]
    assert config.n_latent_dims == 10
    assert config.n_latent_priors == 20
    assert config.n_encoder_layers == 4
    assert config.n_decoder_layers == 2
    assert config.n_progressive_epochs == 5


def test_component_config_name_validation(modality_names):
    config = ModelComponentConfig(
        name="Component With Spaces", modalities=modality_names
    )
    assert config.name == "component_with_spaces"

    # test assignment
    config.name = "Another Name 123"
    assert config.name == "another_name_123"


def test_component_config_missing_required(modality_names):
    # missing name
    with pytest.raises(ValidationError) as excinfo:
        ModelComponentConfig(modalities=modality_names)
    assert "name" in str(excinfo.value)

    # missing modalities
    with pytest.raises(ValidationError) as excinfo:
        ModelComponentConfig(name="TestComp")
    assert "modalities" in str(excinfo.value)


def test_component_config_invalid_types(modality_names):
    base_data = {"name": "TypeTest", "modalities": modality_names}

    # invalid conditioned_on_z type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, conditioned_on_z="not_a_list")

    # invalid conditioned_on_z_hat type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, conditioned_on_z_hat="not_a_list")

    # invalid n_latent_dims type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, n_latent_dims="not_an_int")

    # invalid n_latent_priors type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, n_latent_priors="not_an_int")

    # invalid n_encoder_layers type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, n_encoder_layers="not_an_int")

    # invalid n_decoder_layers type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, n_decoder_layers="not_an_int")  # Matches typo

    # invalid n_progressive_epochs type
    with pytest.raises(ValidationError):
        ModelComponentConfig(**base_data, n_progressive_epochs="not_an_int")

    # invalid modalities type (not list)
    with pytest.raises(ValidationError):
        ModelComponentConfig(name="Test", modalities="not_a_list")

    # invalid item type in modalities list (should be list of strings)
    with pytest.raises(ValidationError):
        ModelComponentConfig(name="Test", modalities=[1, 2, 3])


def test_component_config_assignment(modality_names):
    config = ModelComponentConfig(name="AssignTest", modalities=modality_names[:1])

    assert config.n_latent_dims == 5
    assert config.conditioned_on_z == []
    assert config.modalities == ["rna"]

    config.n_latent_dims = 15
    assert config.n_latent_dims == 15

    config.conditioned_on_z = ["comp_a"]
    assert config.conditioned_on_z == ["comp_a"]

    config.modalities = modality_names
    assert config.modalities == ["rna", "atac"]

    # test invalid assignment
    with pytest.raises(ValidationError):
        config.n_latent_dims = "invalid"

    with pytest.raises(ValidationError):
        config.conditioned_on_z = 123

    # test invalid assignment (list with non-string)
    with pytest.raises(ValidationError):
        config.modalities = ["valid", 123]

    # test invalid assignment (not a list)
    with pytest.raises(ValidationError):
        config.modalities = "not_a_list"
