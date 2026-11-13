import pytest
from pydantic import ValidationError

from cavachon.config import OptimizerConfig, TrainingConfig
from cavachon.config.training_config import EarlyStoppingConfig


def test_defaults():
    t = TrainingConfig()
    assert isinstance(t.optimizer, OptimizerConfig)
    assert t.optimizer.name == "adam"
    assert t.optimizer.learning_rate == pytest.approx(1e-4)
    assert t.max_regular_training_epochs == 500
    assert t.n_parent_annealing_epochs == 1
    assert t.n_kl_annealing_epochs == 25
    assert t.enable_kmeans_init is True
    assert t.learn_z_hat_priors is False
    assert t.kl_annealing_ratio == (0.5, 0.2, 0.3)
    assert t.n_z_hat_annealing_epochs == 0
    assert t.n_z_hat_training_epochs == 0
    assert t.z_hat_kl_weight == 1.0
    assert t.train is True
    assert t.early_stopping is True


def test_optimizer_not_shared():
    a, b = TrainingConfig(), TrainingConfig()
    assert a.optimizer is not b.optimizer
    a.optimizer.learning_rate = 1.0
    assert b.optimizer.learning_rate == pytest.approx(1e-4)


def test_partial_optimizer_fills_defaults():
    t = TrainingConfig(optimizer={"learning_rate": "5e-4"})
    assert t.optimizer.name == "adam"
    assert t.optimizer.learning_rate == pytest.approx(5e-4)


# regression: the default optimizer is no longer injected into the caller's dict
def test_default_optimizer_validator_does_not_mutate_input():
    data = {"train": False}
    t = TrainingConfig.model_validate(data)
    assert data == {"train": False}
    assert t.optimizer.name == "adam"


def test_early_stopping_bool_and_dict():
    assert TrainingConfig(early_stopping=False).early_stopping is False
    es = TrainingConfig(early_stopping={"patience": 3}).early_stopping
    assert isinstance(es, EarlyStoppingConfig)
    assert (es.monitor, es.mode, es.patience) == ("loss", "min", 3)


def test_early_stopping_config_defaults():
    es = EarlyStoppingConfig()
    assert (es.monitor, es.mode, es.patience) == ("loss", "min", 25)


def test_kl_annealing_ratio_length_checked():
    with pytest.raises(ValidationError):
        TrainingConfig(kl_annealing_ratio=[1.0])
    assert TrainingConfig(kl_annealing_ratio=[1, 0, 0]).kl_annealing_ratio == (1.0, 0.0, 0.0)


def test_type_errors():
    with pytest.raises(ValidationError):
        TrainingConfig(max_regular_training_epochs="lots")
    with pytest.raises(ValidationError):
        TrainingConfig(train="maybe")
