import pytest

from cavachon.environment.constants import Constants
from cavachon.layers.modifiers.base.binarize import Binarize
from cavachon.layers.modifiers.base.to_dense import ToDense
from cavachon.layers.modifiers.distribution_modeled_data_presets.independent_bernoulli_modeled_data_modifier import (
    IndependentBernoulliModeledDataModifier,
)


@pytest.fixture
def modality_name():
    return "test_modality"


@pytest.fixture
def modifier(modality_name):
    return IndependentBernoulliModeledDataModifier(modality_name=modality_name)


def test_init(modifier, modality_name):
    assert isinstance(modifier, IndependentBernoulliModeledDataModifier)
    assert modifier.modality_name == modality_name
    assert modifier.modality_key == f"{modality_name}_{Constants.TENSOR_NAME_X_MODEL}"
    assert len(modifier.modifiers) == 2
    assert isinstance(modifier.modifiers[0], ToDense)
    assert isinstance(modifier.modifiers[1], Binarize)
