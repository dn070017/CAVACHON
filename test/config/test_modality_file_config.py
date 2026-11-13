import pytest
import yaml
from pydantic import ValidationError

from cavachon.config import (
    ComponentConfig,
    DatasetConfig,
    FilterConfig,
    ModalityFileConfig,
    ModalityFileFeatureConfig,
    ModalityFileMatrixConfig,
    ModelConfig,
    OptimizerConfig,
    SampleConfig,
    TrainingConfig,
)


def test_matrix_config_defaults_and_required():
    assert ModalityFileMatrixConfig(filename="m.mtx").transpose is False
    with pytest.raises(ValidationError):
        ModalityFileMatrixConfig()


def test_feature_config_defaults():
    f = ModalityFileFeatureConfig(filename="f.tsv")
    assert f.has_headers is False
    assert f.colnames == []


def test_feature_config_colnames_not_shared_between_instances():
    a = ModalityFileFeatureConfig(filename="a")
    b = ModalityFileFeatureConfig(filename="b")
    a.colnames.append("x")
    assert b.colnames == []


def _file_cfg(name="RNA"):
    return {
        "name": name,
        "matrix": {"filename": "m.mtx"},
        "barcodes": {"filename": "b.tsv"},
        "features": {"filename": "f.tsv"},
    }


def test_modality_file_config_nested_parse_and_sanitize():
    m = ModalityFileConfig(**_file_cfg("my rna"))
    assert m.name == "my_rna"
    assert isinstance(m.matrix, ModalityFileMatrixConfig)
    assert isinstance(m.barcodes, ModalityFileFeatureConfig)


@pytest.mark.parametrize("missing", ["name", "matrix", "barcodes", "features"])
def test_modality_file_config_required_fields(missing):
    data = _file_cfg()
    data.pop(missing)
    with pytest.raises(ValidationError):
        ModalityFileConfig(**data)


def test_sample_config_defaults_and_nested():
    s = SampleConfig(name="s1")
    assert s.description == "" and s.modalities == []
    s = SampleConfig(name="s 1", modalities=[_file_cfg()])
    # sample names are NOT sanitized (plain str) unlike modality names
    assert s.name == "s 1"
    assert s.modalities[0].name == "RNA"
    with pytest.raises(ValidationError):
        SampleConfig()
