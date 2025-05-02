import pytest

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.binarize import Binarize
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_observed_data_presets.independent_bernoulli_observed_data_modifier import (
    IndependentBernoulliObservedDataModifier,
)


@pytest.fixture
def modality_name():
    return "test_modality"


@pytest.fixture
def modifier(modality_name):
    return IndependentBernoulliObservedDataModifier(modality_name=modality_name)


def test_init(modifier, modality_name):
    assert isinstance(modifier, IndependentBernoulliObservedDataModifier)
    assert modifier.modality_name == modality_name
    assert (
        modifier.modality_key == f"{modality_name}_{Constants.TENSOR_NAME_X_OBSERVED}"
    )
    assert len(modifier.modifiers) == 2
    assert isinstance(modifier.modifiers[0], ToDense)
    assert isinstance(modifier.modifiers[1], Binarize)


def test_independent_bernoulli_observed_data_modifier_get_config(
    modifier, modality_name
):
    config = modifier.get_config()
    new_modifier = IndependentBernoulliObservedDataModifier.from_config(config)
    assert new_modifier.modality_name == modality_name
    assert (
        new_modifier.modality_key
        == f"{modality_name}_{Constants.TENSOR_NAME_X_OBSERVED}"
    )
    assert len(new_modifier.modifiers) == 2
    assert isinstance(new_modifier.modifiers[0], ToDense)
    assert isinstance(new_modifier.modifiers[1], Binarize)
