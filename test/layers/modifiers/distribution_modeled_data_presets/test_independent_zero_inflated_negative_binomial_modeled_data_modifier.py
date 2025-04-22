import pytest

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.log_transform import LogTransform
from cavachon.layers.modifiers.base.normalize_library_size import NormalizeLibrarySize
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_modeled_data_presets.independent_zero_inflated_negative_binomial_modeled_data_modifier import (
    IndependentZeroInflatedNegativeBinomialModeledDataModifier,
)


@pytest.fixture
def modality_name():
    return "test_modality"


@pytest.fixture
def modifier(modality_name):
    return IndependentZeroInflatedNegativeBinomialModeledDataModifier(
        modality_name=modality_name
    )


def test_init(modifier, modality_name):
    assert isinstance(
        modifier, IndependentZeroInflatedNegativeBinomialModeledDataModifier
    )
    assert modifier.modality_name == modality_name
    assert modifier.modality_key == f"{modality_name}_{Constants.TENSOR_NAME_X_MODEL}"
    assert len(modifier.modifiers) == 3
    assert isinstance(modifier.modifiers[0], ToDense)
    assert isinstance(modifier.modifiers[1], LogTransform)
    assert isinstance(modifier.modifiers[2], NormalizeLibrarySize)
