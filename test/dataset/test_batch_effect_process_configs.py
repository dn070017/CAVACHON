import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from cavachon.dataset.batch_effect_process_configs import BatchEffectProcessConfig


@pytest.fixture
def basic_config():
    return BatchEffectProcessConfig(
        colname="batch", categorical=False, encoder=None, n_vars=1
    )


@pytest.fixture
def categorical_config_1():
    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C"])
    return BatchEffectProcessConfig(
        colname="cell_type", categorical=True, encoder=encoder, n_vars=3
    )


@pytest.fixture
def categorical_config_2():
    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C"])
    return BatchEffectProcessConfig(
        colname="cell_type", categorical=True, encoder=encoder, n_vars=3
    )


@pytest.fixture
def categorical_config_different_classes():
    encoder = LabelEncoder()
    encoder.fit(["X", "Y", "Z"])
    return BatchEffectProcessConfig(
        colname="cell_type", categorical=True, encoder=encoder, n_vars=3
    )


@pytest.fixture
def categorical_config_different_colname():
    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C"])
    return BatchEffectProcessConfig(
        colname="other_type", categorical=True, encoder=encoder, n_vars=3
    )


@pytest.fixture
def categorical_config_different_nvars():
    encoder = LabelEncoder()
    encoder.fit(["A", "B", "C"])
    return BatchEffectProcessConfig(
        colname="cell_type",
        categorical=True,
        encoder=encoder,
        n_vars=4,  # Different n_vars
    )


def test_config_initialization(basic_config, categorical_config_1):
    assert basic_config.colname == "batch"
    assert not basic_config.categorical
    assert basic_config.encoder is None
    assert basic_config.n_vars == 1

    assert categorical_config_1.colname == "cell_type"
    assert categorical_config_1.categorical
    assert isinstance(categorical_config_1.encoder, LabelEncoder)
    assert categorical_config_1.n_vars == 3
    np.testing.assert_array_equal(
        categorical_config_1.encoder.classes_, ["A", "B", "C"]
    )


def test_eq_identical_configs(categorical_config_1, categorical_config_2):
    assert categorical_config_1 == categorical_config_2


def test_eq_different_colname(
    categorical_config_1, categorical_config_different_colname
):
    assert categorical_config_1 != categorical_config_different_colname


def test_eq_different_categorical(basic_config, categorical_config_1):
    basic_config_modified = BatchEffectProcessConfig(
        colname="cell_type",
        categorical=False,  # Different categorical
        encoder=None,
        n_vars=3,
    )
    assert basic_config_modified != categorical_config_1


def test_eq_different_nvars(categorical_config_1, categorical_config_different_nvars):
    assert categorical_config_1 != categorical_config_different_nvars


def test_eq_one_encoder_none(basic_config, categorical_config_1):
    basic_config_modified = BatchEffectProcessConfig(
        colname="cell_type",
        categorical=True,
        encoder=None,  # One encoder is None
        n_vars=3,
    )
    assert basic_config_modified != categorical_config_1
    assert categorical_config_1 != basic_config_modified  # Test symmetry


def test_eq_both_encoders_none(basic_config):
    config_copy = BatchEffectProcessConfig(
        colname="batch", categorical=False, encoder=None, n_vars=1
    )
    assert basic_config == config_copy


def test_eq_different_encoder_classes(
    categorical_config_1, categorical_config_different_classes
):
    assert categorical_config_1 != categorical_config_different_classes


def test_eq_non_config_object(categorical_config_1):
    with pytest.raises(NotImplementedError):
        categorical_config_1 == "not a config object"

    with pytest.raises(NotImplementedError):
        "not a config object" == categorical_config_1


def test_eq_different_types():
    class Dummy:
        pass

    config = BatchEffectProcessConfig("col", False, None, 1)
    dummy_obj = Dummy()

    with pytest.raises(NotImplementedError):
        config == dummy_obj

    with pytest.raises(NotImplementedError):
        dummy_obj == config
