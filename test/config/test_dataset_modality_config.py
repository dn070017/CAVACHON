import pytest
from pydantic import ValidationError

from cavachon.config.dataset_modality_config import DatasetModalityConfig
from cavachon.utils.general_utils import GeneralUtils


def test_modality_config_valid():
    config_data = {
        "name": "RNA",
        "dist": "IndependentZeroInflatedNegativeBinomial",
        "h5ad": "rna.h5ad",
        "batch_effect_colnames": ["batch", "donor"],
    }
    config = DatasetModalityConfig(**config_data)

    assert config.name == GeneralUtils.convert_to_tensorflow_compatible_string("rna")
    assert config.dist == "IndependentZeroInflatedNegativeBinomial"
    assert config.h5ad == "rna.h5ad"
    assert config.batch_effect_colnames == ["batch", "donor"]


def test_modality_config_defaults():
    config_data = {
        "name": "ATAC",
        "dist": "IndependentBernoulli",
    }
    config = DatasetModalityConfig(**config_data)

    assert config.name == GeneralUtils.convert_to_tensorflow_compatible_string("atac")
    assert config.dist == "IndependentBernoulli"
    assert config.h5ad is None
    assert config.batch_effect_colnames == []


def test_modality_config_name_validation():
    config = DatasetModalityConfig(name="My Modality Name", dist="IndependentBernoulli")
    assert config.name == "my_modality_name"

    config.name = "Another Name With Spaces"
    assert config.name == "another_name_with_spaces"


def test_modality_config_invalid_dist():
    config_data = {
        "name": "InvalidDistModality",
        "dist": "InvalidDistribution",
    }
    with pytest.raises(ValidationError):
        DatasetModalityConfig(**config_data)


def test_modality_config_assignment():
    config = DatasetModalityConfig(
        name="TestModality", dist="MultivariateNormalDiagDistribution"
    )
    assert config.h5ad is None
    assert config.batch_effect_colnames == []

    config.h5ad = "test.h5ad"
    assert config.h5ad == "test.h5ad"

    config.batch_effect_colnames = ["new_batch"]
    assert config.batch_effect_colnames == ["new_batch"]

    # Test invalid assignment (wrong type)
    with pytest.raises(ValidationError):
        config.batch_effect_colnames = "invalid_type"

    # Test invalid assignment (wrong dist literal)
    with pytest.raises(ValidationError):
        config.dist = "InvalidDist"
