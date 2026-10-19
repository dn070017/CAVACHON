import operator
from typing import Callable, Dict, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


ThresholdOperator = Literal["leq", "geq", "lt", "gt"]

THRESHOLD_OPERATORS: Dict[str, Callable[..., pd.Series]] = {
    "leq": operator.le,
    "geq": operator.ge,
    "lt": operator.lt,
    "gt": operator.gt,
}

THRESHOLD_OPERATOR_SYMBOLS: Dict[str, str] = {
    "leq": "<=",
    "geq": ">=",
    "lt": "<",
    "gt": ">",
}


class DataFrameUtils:
    """DataFrameUtils

    Class containing multiple utility functions for pd.DataFrame or
    pd.Series.

    """

    @staticmethod
    def apply_threshold(
        data: pd.Series,
        threshold: float,
        operator: ThresholdOperator,
    ) -> pd.Series:
        """Return a boolean mask comparing ``data`` to ``threshold``.

        Parameters
        ----------
        data: pd.Series
            values to compare.

        threshold: float
            threshold value.

        operator: ThresholdOperator
            one of ``leq``, ``geq``, ``lt``, ``gt``.

        Returns
        -------
        pd.Series:
            boolean mask with the comparison result.

        """
        comparison = THRESHOLD_OPERATORS.get(operator)
        if comparison is None:
            raise ValueError(f"Unknown threshold operator: {operator}")
        return comparison(data, threshold)

    @staticmethod
    def filter_by_threshold(
        data: pd.Series,
        threshold: float,
        operator: ThresholdOperator,
        use_absolute: bool = False,
    ) -> pd.Series:
        """Return a boolean mask after optional absolute transformation.

        Parameters
        ----------
        data: pd.Series
            values to compare.

        threshold: float
            threshold value.

        operator: ThresholdOperator
            one of ``leq``, ``geq``, ``lt``, ``gt``.

        use_absolute: bool, optional
            whether to compare ``|data|`` to ``threshold``. Defaults to
            ``False``.

        Returns
        -------
        pd.Series:
            boolean mask with the comparison result.

        """
        values = data.abs() if use_absolute else data
        return DataFrameUtils.apply_threshold(values, threshold, operator)

    @staticmethod
    def check_is_categorical(data: pd.Series, threshold: float = 0.2) -> bool:
        """Check if the given Series variable is categorical. This
        function computes the proportion of the number of unique
        values, if the number of unique values is larger than
        `threshold`, it is considered as a continuous variable.

        Parameters
        ----------
        data: pd.Series
            pd.Series needs to be check.

        threshold: float
            threshold of proportion of the number of unique values to
            determine if one variable is categorical or not. Defaults
            to 0.2.

        Returns
        -------
        bool:
            True if the series is a categorical variable, otherwise
            False.
        """
        n_obs = data.size
        return len(np.unique(data)) < n_obs * threshold or not is_numeric_dtype(data)
