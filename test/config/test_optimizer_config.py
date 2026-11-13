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


def test_optimizer_defaults():
    o = OptimizerConfig()
    assert o.name == "adam"
    assert o.learning_rate == pytest.approx(1e-4)


def test_optimizer_learning_rate_from_yaml_string():
    # PyYAML (YAML 1.1) parses ``5e-4`` as a *string*; the validator
    # must coerce it.
    raw = yaml.load("learning_rate: 5e-4", Loader=yaml.FullLoader)
    assert raw["learning_rate"] == "5e-4"
    o = OptimizerConfig(**raw)
    assert isinstance(o.learning_rate, float)
    assert o.learning_rate == pytest.approx(5e-4)


def test_optimizer_learning_rate_int():
    assert OptimizerConfig(learning_rate=1).learning_rate == 1.0


def test_optimizer_learning_rate_invalid_string():
    with pytest.raises(ValidationError):
        OptimizerConfig(learning_rate="fast")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: float(None) raises raw TypeError in before-validator "
    "instead of a pydantic ValidationError",
)
def test_optimizer_learning_rate_none_raises_validation_error():
    with pytest.raises(ValidationError):
        OptimizerConfig(learning_rate=None)
