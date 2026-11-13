import pytest
import yaml
from pydantic import ValidationError

from cavachon.config import (
    OptimizerConfig,
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


# regression: learning_rate None raises ValidationError, not TypeError
@pytest.mark.parametrize("value", [None, [1e-4], {"lr": 1}])
def test_optimizer_learning_rate_none_raises_validation_error(value):
    with pytest.raises(ValidationError, match="learning_rate should be a number"):
        OptimizerConfig(learning_rate=value)
