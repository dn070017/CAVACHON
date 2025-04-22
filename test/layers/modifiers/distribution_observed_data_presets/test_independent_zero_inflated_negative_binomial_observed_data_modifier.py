import pytest

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_observed_data_presets.independent_zero_inflated_negative_binomial_observed_data_modifier import (
    IndependentZeroInflatedNegativeBinomialObservedDataModifier,
)


@pytest.fixture
def modality_name():
    return "test_modality"


@pytest.fixture
def modifier(modality_name):
    return IndependentZeroInflatedNegativeBinomialObservedDataModifier(
        modality_name=modality_name
    )


def test_init(modifier, modality_name):
    assert isinstance(
        modifier, IndependentZeroInflatedNegativeBinomialObservedDataModifier
    )
    assert modifier.modality_name == modality_name
    assert (
        modifier.modality_key == f"{modality_name}_{Constants.TENSOR_NAME_X_OBSERVED}"
    )
    assert len(modifier.modifiers) == 1
    assert isinstance(modifier.modifiers[0], ToDense)
