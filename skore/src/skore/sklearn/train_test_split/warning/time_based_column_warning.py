"""'Time-based column' warning.

This warning is shown when ``X`` contains a time-based column.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Union

from skore.sklearn.train_test_split.warning.train_test_split_warning import (
    TrainTestSplitWarning,
)

if TYPE_CHECKING:
    ArrayLike = Any


class TimeBasedColumnWarning(TrainTestSplitWarning):
    """Check whether ``X`` contains a time-based column."""

    @staticmethod
    def _MSG(df_name, offending_column_name):
        column_info = (
            f"(column {offending_column_name}, dataframe {offending_column_name})"
            if df_name is not None
            else f"(column {offending_column_name})"
        )

        return (
            f"We detected a time-based column {column_info} in your data. "
            "We recommend using TimeSeriesSplit instead of train_test_split. "
            "Otherwise you might train on future data to predict the past, or get "
            "inflated model performance evaluation because natural drift will not be "
            "taken into account."
        )

    @staticmethod
    def check(X: ArrayLike, **kwargs) -> Union[str, None]:
        """Check whether ``X`` contains a time-based column.

        Parameters
        ----------
        X : array-like or None
            A data matrix that can contain datetime data, e.g. a DataFrame.

        Returns
        -------
        warning
            None if the check passed, otherwise the warning message.
        """
        if not hasattr(X, "columns"):
            return None

        dtypes = [(col, X[col].dtype) for col in X.columns]
        datetime_columns = [
            col
            for col, dtype in dtypes
            if re.search("date", str(type(dtype)), flags=re.IGNORECASE)
        ]
        if datetime_columns:
            df_name = X.name if hasattr(X, "name") else None
            return TimeBasedColumnWarning._MSG(df_name, datetime_columns[0])
