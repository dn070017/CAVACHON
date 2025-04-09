import pandas as pd

from cavachon.utils.dataframe_utils import DataFrameUtils


def test_check_is_categorical_positive_numeric():
    data = pd.DataFrame(
        {"Test": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    )
    assert DataFrameUtils.check_is_categorical(data["Test"])


def test_check_is_categorical_positive_categorical():
    data = pd.DataFrame(
        {"Test": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"]}
    )
    assert DataFrameUtils.check_is_categorical(data["Test"])


def test_check_is_categorical_negative_numeric():
    data = pd.DataFrame({"Test": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert not DataFrameUtils.check_is_categorical(data["Test"])


def test_check_is_categorical_negative_categorical():
    data = pd.DataFrame({"Test": ["A", "B", "C", "D", "E"]})
    assert DataFrameUtils.check_is_categorical(data["Test"])


def test_check_is_categorical_positive_threshold():
    data = pd.DataFrame(
        {
            "Test": [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ]
        }
    )
    assert DataFrameUtils.check_is_categorical(data["Test"], 0.5)


def test_check_is_categorical_negative_threshold():
    data = pd.DataFrame(
        {
            "Test": [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ]
        }
    )
    assert not DataFrameUtils.check_is_categorical(data["Test"], 0.1)
