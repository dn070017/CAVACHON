import numpy as np
import pandas as pd
import pytest

from cavachon.utils import DataFrameUtils
from cavachon.utils.dataframe_utils import (
    THRESHOLD_OPERATOR_SYMBOLS,
    THRESHOLD_OPERATORS,
)


@pytest.fixture
def series():
    return pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=list("abcde"))


def test_operator_tables_have_same_keys():
    assert set(THRESHOLD_OPERATORS) == set(THRESHOLD_OPERATOR_SYMBOLS) == {"leq", "geq", "lt", "gt"}


@pytest.mark.parametrize(
    "op,expected",
    [
        ("leq", [True, True, True, False, False]),
        ("geq", [False, False, True, True, True]),
        ("lt", [True, True, False, False, False]),
        ("gt", [False, False, False, True, True]),
    ],
)
def test_apply_threshold(series, op, expected):
    mask = DataFrameUtils.apply_threshold(series, 0.0, op)
    assert mask.tolist() == expected
    assert mask.dtype == bool
    assert mask.index.equals(series.index)


def test_apply_threshold_unknown_operator(series):
    with pytest.raises(ValueError, match="Unknown threshold operator"):
        DataFrameUtils.apply_threshold(series, 0.0, "eq")


def test_apply_threshold_nan_is_false():
    mask = DataFrameUtils.apply_threshold(pd.Series([np.nan, 1.0]), 0.0, "geq")
    assert mask.tolist() == [False, True]


def test_apply_threshold_empty():
    mask = DataFrameUtils.apply_threshold(pd.Series([], dtype=float), 0.0, "gt")
    assert len(mask) == 0


def test_filter_by_threshold_absolute(series):
    mask = DataFrameUtils.filter_by_threshold(series, 1.5, "geq", use_absolute=True)
    assert mask.tolist() == [True, False, False, False, True]


def test_filter_by_threshold_default_not_absolute(series):
    mask = DataFrameUtils.filter_by_threshold(series, 1.5, "geq")
    assert mask.tolist() == [False, False, False, False, True]


def test_filter_by_threshold_does_not_mutate_input(series):
    before = series.copy()
    DataFrameUtils.filter_by_threshold(series, 1.0, "lt", use_absolute=True)
    pd.testing.assert_series_equal(series, before)


def test_filter_by_threshold_unknown_operator(series):
    with pytest.raises(ValueError):
        DataFrameUtils.filter_by_threshold(series, 1.0, "neq", use_absolute=True)


class TestCheckIsCategorical:
    def test_low_cardinality_numeric_is_categorical(self):
        s = pd.Series([0.0] * 6 + [1.0] * 6)
        assert DataFrameUtils.check_is_categorical(s)

    def test_high_cardinality_numeric_is_continuous(self):
        s = pd.Series(np.arange(12, dtype=float))
        assert not DataFrameUtils.check_is_categorical(s)

    def test_string_always_categorical(self):
        s = pd.Series([f"s{i}" for i in range(10)])
        assert DataFrameUtils.check_is_categorical(s)

    def test_pandas_categorical_dtype_is_categorical(self):
        s = pd.Series(pd.Categorical(list(range(10))))
        assert DataFrameUtils.check_is_categorical(s)

    def test_threshold_is_strict(self):
        # 2 unique in 10 obs: 2 < 10 * 0.2 == 2 is False -> continuous
        s = pd.Series([0, 1] * 5)
        assert not DataFrameUtils.check_is_categorical(s)
        assert DataFrameUtils.check_is_categorical(s, threshold=0.3)

    def test_custom_threshold(self):
        s = pd.Series(np.arange(10, dtype=float))
        assert DataFrameUtils.check_is_categorical(s, threshold=1.01)

    def test_empty_numeric_series_is_continuous(self):
        assert not DataFrameUtils.check_is_categorical(pd.Series([], dtype=float))

    def test_nan_counted_once(self):
        s = pd.Series([np.nan] * 10 + [1.0] * 10)
        assert DataFrameUtils.check_is_categorical(s)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="BUG: np.unique on object Series mixing str and NaN raises TypeError",
    )
    def test_object_with_missing_values(self):
        s = pd.Series(["a", "b", np.nan, "a"], dtype=object)
        assert DataFrameUtils.check_is_categorical(s)

    def test_boolean_small_sample_is_treated_as_continuous(self):
        # documents heuristic: bool is numeric, so with n_obs <= 10 a
        # 2-level boolean column falls on the "continuous" side.
        assert not DataFrameUtils.check_is_categorical(pd.Series([True, False] * 5))
        assert DataFrameUtils.check_is_categorical(pd.Series([True, False] * 6))
