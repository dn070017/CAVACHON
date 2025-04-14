import pytest
from pydantic import ValidationError

from cavachon.config.dataset_config import DatasetConfig
from cavachon.config.dataset_modality_config import (
    DatasetModalityConfig,  # Renamed from modality_config
)


def test_dataset_config_defaults():
    modality_data = {"name": "RNA", "dist": "IndependentBernoulli"}
    modality_config = DatasetModalityConfig(**modality_data)
    dataset_config_data = {"modalities": [modality_config]}

    config = DatasetConfig(**dataset_config_data)

    assert config.batch_size == 128
    assert config.shuffle is False
    assert len(config.modalities) == 1
    assert isinstance(config.modalities[0], DatasetModalityConfig)
    assert config.modalities[0].name == "rna"

    dataset_config_data = {"modalities": [modality_data]}
    assert len(config.modalities) == 1
    assert isinstance(config.modalities[0], DatasetModalityConfig)
    assert config.modalities[0].name == "rna"


def test_dataset_config_custom_values():
    modality_data_1 = {"name": "ATAC", "dist": "IndependentBernoulli"}
    modality_data_2 = {
        "name": "RNA",
        "dist": "IndependentZeroInflatedNegativeBinomial",
    }
    dataset_config_data = {
        "batch_size": 64,
        "shuffle": True,
        "modalities": [modality_data_1, modality_data_2],
    }

    config = DatasetConfig(**dataset_config_data)

    assert config.batch_size == 64
    assert config.shuffle is True
    assert len(config.modalities) == 2
    assert isinstance(config.modalities[0], DatasetModalityConfig)
    assert isinstance(config.modalities[1], DatasetModalityConfig)
    assert config.modalities[0].name == "atac"
    assert config.modalities[1].name == "rna"


def test_dataset_config_missing_modalities():
    dataset_config_data = {
        "batch_size": 256,
        "shuffle": False,
        # modalities is missing
    }
    with pytest.raises(ValidationError) as excinfo:
        DatasetConfig(**dataset_config_data)
    assert "modalities" in str(excinfo.value)


def test_dataset_config_invalid_types():
    modality_data = {"name": "Test", "dist": "IndependentBernoulli"}

    # invalid batch_size type
    with pytest.raises(ValidationError):
        DatasetConfig(
            batch_size="not_an_int", shuffle=False, modalities=[modality_data]
        )

    # invalid shuffle type
    with pytest.raises(ValidationError):
        DatasetConfig(batch_size=128, shuffle="not_a_bool", modalities=[modality_data])

    # invalid modalities type (not a list)
    with pytest.raises(ValidationError):
        DatasetConfig(batch_size=128, shuffle=False, modalities=modality_data)

    # invalid item type in modalities list
    with pytest.raises(ValidationError):
        DatasetConfig(
            batch_size=128, shuffle=False, modalities=["not_a_modality_config"]
        )


def test_dataset_config_assignment():
    modality_1 = DatasetModalityConfig(name="M1", dist="IndependentBernoulli")
    modality_2 = DatasetModalityConfig(
        name="M2", dist="IndependentZeroInflatedNegativeBinomial"
    )
    config = DatasetConfig(modalities=[modality_1])

    assert config.batch_size == 128
    assert config.shuffle is False

    config.batch_size = 32
    assert config.batch_size == 32

    config.shuffle = True
    assert config.shuffle is True

    config.modalities = [modality_1, modality_2]
    assert len(config.modalities) == 2
    assert config.modalities[0].name == "m1"
    assert config.modalities[1].name == "m2"

    # Test invalid assignment
    with pytest.raises(ValidationError):
        config.batch_size = "invalid"

    with pytest.raises(ValidationError):
        config.shuffle = 123

    with pytest.raises(ValidationError):
        config.modalities = "not_a_list"
