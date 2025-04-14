import warnings

import pytest
from cavachon.config.training_config import TrainingConfig
from cavachon.config.training_optimizer_config import (
    AdamOptimizerParams,
    TrainingOptimizerConfig,
)
from pydantic import ValidationError


def test_training_config_defaults():
    config = TrainingConfig()
    assert isinstance(config.optimizer, TrainingOptimizerConfig)
    assert config.optimizer.kind == "adam"
    assert config.max_n_epochs == 500
    assert config.train is True
    assert config.early_stopping is True
    assert config.save_weight is True


def test_training_config_custom_values():
    optimizer_data = {
        "kind": "adamw",
        "params": {"name": "adamw", "learning_rate": 0.005},
    }
    config_data = {
        "optimizer": optimizer_data,
        "max_n_epochs": 100,
        "train": False,
        "early_stopping": False,
        "save_weight": False,
    }
    config = TrainingConfig(**config_data)

    assert isinstance(config.optimizer, TrainingOptimizerConfig)
    assert config.optimizer.kind == "adamw"
    assert config.optimizer.params.learning_rate == 0.005
    assert config.max_n_epochs == 100
    assert config.train is False
    assert config.early_stopping is False
    assert config.save_weight is False


def test_training_config_validation_warnings():
    with pytest.warns(UserWarning):
        TrainingConfig(train=False, early_stopping=True, save_weight=True)

    with pytest.warns(UserWarning):
        TrainingConfig(train=False, early_stopping=False, save_weight=True)

    with pytest.warns(UserWarning):
        TrainingConfig(train=False, early_stopping=True, save_weight=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # raise error if any warning occurs
        TrainingConfig(train=True, early_stopping=True, save_weight=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # raise error if any warning occurs
        TrainingConfig(train=False, early_stopping=False, save_weight=False)


def test_training_config_invalid_types():
    # invalid optimizer type
    with pytest.raises(ValidationError):
        TrainingConfig(optimizer="not_an_optimizer_config")

    # invalid max_n_epochs type
    with pytest.raises(ValidationError):
        TrainingConfig(max_n_epochs="not_an_int")

    # invalid train type
    with pytest.raises(ValidationError):
        TrainingConfig(train="not_a_bool")

    # invalid early_stopping type
    with pytest.raises(ValidationError):
        TrainingConfig(early_stopping="not_a_bool")

    # invalid save_weight type
    with pytest.raises(ValidationError):
        TrainingConfig(save_weight="not_a_bool")


def test_training_config_assignment():
    config = TrainingConfig()
    assert config.max_n_epochs == 500
    assert config.train is True
    assert isinstance(config.optimizer.params, AdamOptimizerParams)

    config.max_n_epochs = 200
    assert config.max_n_epochs == 200

    with pytest.warns(UserWarning):
        config.train = False
        assert config.train is False

    with pytest.warns(UserWarning):
        config.early_stopping = True
        assert config.early_stopping is True

    with pytest.warns(UserWarning):
        config.save_weight = True
        assert config.save_weight is True

    config.train = True
    config.optimizer = TrainingOptimizerConfig(kind="lion")
    assert config.optimizer.kind == "lion"

    with pytest.raises(ValidationError):
        config.max_n_epochs = "invalid"

    with pytest.raises(ValidationError):
        config.train = 123

    with pytest.raises(ValidationError):
        config.optimizer = "invalid"
