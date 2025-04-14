import pytest
from pydantic import ValidationError

from cavachon.config.training_optimizer_config import (
    AdafactorOptimizerParams,
    AdamOptimizerParams,
    AdamWOptimizerParams,
    LionOptimizerParams,
    TrainingOptimizerConfig,
)


def test_optimizer_config_default_adam():
    config = TrainingOptimizerConfig()
    assert config.kind == "adam"
    assert isinstance(config.params, AdamOptimizerParams)
    assert config.params.name == "adam"
    assert config.params.learning_rate == 0.001
    assert config.params.beta_1 == 0.9
    assert config.params.beta_2 == 0.999
    assert config.params.epsilon == 1e-07
    assert config.params.amsgrad is False
    assert config.params.weight_decay is None
    assert config.params.jit_compile is True


def test_optimizer_config_explicit_adam_with_defaults():
    config = TrainingOptimizerConfig(kind="adam")
    assert config.kind == "adam"
    assert isinstance(config.params, AdamOptimizerParams)
    assert config.params.name == "adam"
    assert config.params.learning_rate == 0.001


def test_optimizer_config_explicit_adam_with_custom_params():
    config = TrainingOptimizerConfig(
        kind="adam",
        params={"name": "adam", "learning_rate": 0.01, "beta_1": 0.8},
    )
    assert config.kind == "adam"
    assert isinstance(config.params, AdamOptimizerParams)
    assert config.params.name == "adam"
    assert config.params.learning_rate == 0.01
    assert config.params.beta_1 == 0.8
    assert config.params.beta_2 == 0.999
    assert config.params.jit_compile is True


def test_optimizer_config_adamw_defaults():
    config = TrainingOptimizerConfig(kind="adamw")
    assert config.kind == "adamw"
    assert isinstance(config.params, AdamWOptimizerParams)
    assert config.params.name == "adamw"
    assert config.params.learning_rate == 0.001
    assert config.params.beta_1 == 0.9
    assert config.params.beta_2 == 0.999
    assert config.params.epsilon == 1e-07
    assert config.params.amsgrad is False
    assert config.params.weight_decay == 0.004
    assert config.params.jit_compile is True


def test_optimizer_config_adamw_custom_params():
    config = TrainingOptimizerConfig(
        kind="adamw",
        params={"name": "adamw", "weight_decay": 0.01, "jit_compile": False},
    )
    assert config.kind == "adamw"
    assert isinstance(config.params, AdamWOptimizerParams)
    assert config.params.name == "adamw"
    assert config.params.learning_rate == 0.001
    assert config.params.weight_decay == 0.01
    assert config.params.jit_compile is False


def test_optimizer_config_adafactor_defaults():
    config = TrainingOptimizerConfig(kind="adafactor")
    assert config.kind == "adafactor"
    assert isinstance(config.params, AdafactorOptimizerParams)
    assert config.params.name == "adafactor"
    assert config.params.learning_rate == 0.001
    assert config.params.beta_2_decay == -0.8
    assert config.params.epsilon_1 == 1e-30
    assert config.params.epsilon_2 == 0.001
    assert config.params.clip_threshold == 1.0
    assert config.params.relative_step is True
    assert config.params.jit_compile is True


def test_optimizer_config_adafactor_custom_params():
    config = TrainingOptimizerConfig(
        kind="adafactor",
        params={"name": "adafactor", "learning_rate": 0.005, "relative_step": False},
    )
    assert config.kind == "adafactor"
    assert isinstance(config.params, AdafactorOptimizerParams)
    assert config.params.name == "adafactor"
    assert config.params.learning_rate == 0.005
    assert config.params.relative_step is False
    assert config.params.beta_2_decay == -0.8


def test_optimizer_config_lion_defaults():
    config = TrainingOptimizerConfig(kind="lion")
    assert config.kind == "lion"
    assert isinstance(config.params, LionOptimizerParams)
    assert config.params.name == "lion"
    assert config.params.learning_rate == 0.001
    assert config.params.beta_1 == 0.9
    assert config.params.beta_2 == 0.999
    assert config.params.jit_compile is True


def test_optimizer_config_lion_custom_params():
    config = TrainingOptimizerConfig(
        kind="lion",
        params={"name": "lion", "beta_1": 0.95, "learning_rate": 1e-4},
    )
    assert config.kind == "lion"
    assert isinstance(config.params, LionOptimizerParams)
    assert config.params.name == "lion"
    assert config.params.learning_rate == 1e-4
    assert config.params.beta_1 == 0.95
    assert config.params.beta_2 == 0.999


def test_optimizer_config_invalid_name():
    with pytest.raises(ValidationError) as excinfo:
        TrainingOptimizerConfig(kind="invalid_optimizer")
    assert "Input should be 'adafactor', 'adam', 'adamw' or 'lion'" in str(
        excinfo.value
    )


def test_optimizer_config_mismatched_params_name():
    with pytest.warns(UserWarning):
        config = TrainingOptimizerConfig(
            kind="adam",
            params={"name": "lion", "learning_rate": 0.05},
        )
    assert config.kind == "adam"
    assert isinstance(config.params, LionOptimizerParams)
    assert config.params.name == "lion"
    assert config.params.learning_rate == 0.05
    assert config.params.beta_1 == 0.9


def test_optimizer_config_invalid_param_type():
    with pytest.raises(ValidationError):
        TrainingOptimizerConfig(
            kind="adam", params={"name": "adam", "learning_rate": "not_a_float"}
        )

    with pytest.raises(ValidationError):
        TrainingOptimizerConfig(
            kind="adamw", params={"name": "adamw", "weight_decay": "not_a_float"}
        )

    with pytest.raises(ValidationError):
        TrainingOptimizerConfig(
            kind="adafactor",
            params={"name": "adafactor", "relative_step": "not_a_bool"},
        )

    with pytest.raises(ValidationError):
        TrainingOptimizerConfig(
            kind="lion", params={"name": "lion", "beta_1": "not_a_float"}
        )


def test_optimizer_config_assignment():
    config = TrainingOptimizerConfig(kind="adam")
    assert isinstance(config.params, AdamOptimizerParams)
    assert config.params.learning_rate == 0.001

    with pytest.warns(UserWarning):
        config.params = AdamWOptimizerParams(learning_rate=0.02, weight_decay=0.05)
    assert config.kind == "adam"
    assert isinstance(config.params, AdamWOptimizerParams)
    assert config.params.name == "adamw"
    assert config.params.learning_rate == 0.02
    assert config.params.weight_decay == 0.05

    with pytest.warns(UserWarning):
        config.params = {"name": "lion", "beta_1": 0.88}
    assert isinstance(config.params, LionOptimizerParams)
    assert config.params.name == "lion"
    assert config.params.beta_1 == 0.88
    assert config.params.learning_rate == 0.001

    # Test invalid assignment
    with pytest.raises(ValidationError):
        config.params = {"name": "adam", "learning_rate": "invalid"}

    # Assigning a non-dict/non-model value
    with pytest.raises(ValidationError):
        config.params = "not_valid_params"
