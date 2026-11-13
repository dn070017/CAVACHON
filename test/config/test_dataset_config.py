import pytest
from pydantic import ValidationError

from cavachon.config import (
    DatasetConfig,
)


def test_dataset_defaults():
    d = DatasetConfig()
    assert d.batch_size == 128
    assert d.shuffle is False


def test_dataset_coercion_and_errors():
    d = DatasetConfig(batch_size="64", shuffle="true")
    assert d.batch_size == 64 and d.shuffle is True
    with pytest.raises(ValidationError):
        DatasetConfig(batch_size="many")
    with pytest.raises(ValidationError):
        DatasetConfig(batch_size=1.5)
