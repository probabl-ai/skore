"""'Time-based column' warning.

This warning is shown when ``X`` contains a time-based column.
Currently, only pandas and polars DataFrames are supported.
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
    """Check whether the design matrix ``X`` contains a time-based column.

    Currently, only pandas and polars DataFrames are supported.
    """

    @staticmethod
    def _MSG(df_name: Union[str, None], offending_column_names: list[str]) -> str:
        s = "" if len(offending_column_names) == 1 else "s"

        df_name_info = "" if df_name is None else f', dataframe "{df_name}"'

        column_info = f"(column{s} {', '.join(offending_column_names)}{df_name_info})"

        return (
            f"We detected some time-based columns {column_info} in your data. "
            "We recommend using scikit-learn's TimeSeriesSplit instead of "
            "train_test_split. Otherwise you might train on future data to "
            "predict the past, or get inflated model performance evaluation "
            "because natural drift will not be taken into account."
        )

    @staticmethod
    def check(X: ArrayLike, **kwargs) -> Union[str, None]:
        """Check whether the design matrix ``X`` contains a time-based column.

        Currently, only pandas and polars DataFrames are supported.

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
        # NOTE: Searching by regex allows us to avoid depending explicitly
        # on Pandas or Polars
        datetime_columns = [
            f'"{col}"'
            for col, dtype in dtypes
            if re.search("date", str(type(dtype)), flags=re.IGNORECASE)
        ]
        if datetime_columns:
            df_name = X.name if hasattr(X, "name") else None
            return TimeBasedColumnWarning._MSG(df_name, datetime_columns)

        return None
