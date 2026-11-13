
from cavachon.config import (
    ComponentConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)


def test_model_config_defaults():
    m = ModelConfig()
    assert m.name == "cavachon"
    assert m.components == []
    assert isinstance(m.training, TrainingConfig)
    assert isinstance(m.dataset, DatasetConfig)
    assert m.load_weights is False
    assert m.save_weights is True


def test_model_config_defaults_not_shared():
    a, b = ModelConfig(), ModelConfig()
    assert a.training is not b.training
    assert a.dataset is not b.dataset
    a.components.append(ComponentConfig(name="x"))
    assert b.components == []


def test_model_config_parses_components_and_sanitizes_name():
    m = ModelConfig(
        name="my model",
        components=[{"name": "c", "modalities": [{"name": "RNA"}]}],
    )
    assert m.name == "my_model"
    assert m.components[0].modality_names == ["RNA"]
