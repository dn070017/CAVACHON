import re

import pytest

from cavachon.environment import Constants
from cavachon.environment.constants import Constants as ConstantsDirect


def test_constants_reexported_from_package():
    assert Constants is ConstantsDirect


def test_tensor_names():
    assert Constants.TENSOR_NAME_X == "matrix"
    assert Constants.TENSOR_NAME_ORIGIONAL_X == "_matrix"
    assert Constants.TENSOR_NAME_LIBSIZE == "libsize"
    assert Constants.TENSOR_NAME_BATCH == "batch_effect"


@pytest.mark.parametrize(
    "name", ["abc", "a1", "1a", ".a", "a_b", "a/b", "a\\b", "a>b", "a-b", "A.b"]
)
def test_tensorflow_name_regex_accepts_valid(name):
    assert re.match(Constants.TENSORFLOW_NAME_REGEX, name)


@pytest.mark.parametrize("name", ["", "_a", "-a", "/a", "a b", "a:b", "a*b", "é"])
def test_tensorflow_name_regex_rejects_invalid(name):
    assert re.fullmatch(Constants.TENSORFLOW_NAME_REGEX, name) is None


@pytest.mark.parametrize("name,ok", [("a", True), ("1", True), (".", True), ("_", False), ("", False)])
def test_tensorflow_name_start_regex(name, ok):
    assert bool(re.match(Constants.TENSORFLOW_NAME_START_REGEX, name)) is ok


def test_supported_modalities_and_default_distributions_consistent():
    assert Constants.SUPPORTED_MODALITY_TYPES == {"atac", "rna"}
    assert set(Constants.DEFAULT_MODALITY_DISTRIBUTION) == Constants.SUPPORTED_MODALITY_TYPES
    assert Constants.DEFAULT_MODALITY_DISTRIBUTION["atac"] == "IndependentBernoulli"
    assert (
        Constants.DEFAULT_MODALITY_DISTRIBUTION["rna"]
        == "IndependentZeroInflatedNegativeBinomial"
    )


def test_required_field_lists_reference_other_constants():
    assert Constants.CONFIG_FIELD_MODALITY_REQUIRED == ["name", Constants.CONFIG_FIELD_MODALITY_TYPE]
    assert Constants.CONFIG_FIELD_SAMPLE_REQUIRED == ["name", Constants.CONFIG_FIELD_MODALITY]
    assert Constants.CONFIG_FIELD_MODEL_REQUIRED == [Constants.CONFIG_FIELD_MODEL_COMPONENT]
    assert Constants.CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED == [
        Constants.CONFIG_FIELD_SAMPLE_MODALITY_MTX,
        Constants.CONFIG_FIELD_SAMPLE_MODALITY_OBS,
        Constants.CONFIG_FIELD_SAMPLE_MODALITY_VAR,
    ]


def test_condition_field_names():
    assert Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z == "conditioned_on_z"
    assert Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT == "conditioned_on_z_hat"


def test_all_string_constants_are_nonempty_and_model_outputs_unique():
    str_values = {
        k: v for k, v in vars(Constants).items() if not k.startswith("__") and isinstance(v, str)
    }
    assert all(len(v) > 0 for v in str_values.values())
    outputs = [v for k, v in str_values.items() if k.startswith("MODEL_OUTPUTS_")]
    assert len(outputs) == len(set(outputs))
    modules = [v for k, v in str_values.items() if k.startswith("MODULE_")]
    assert len(modules) == len(set(modules))
