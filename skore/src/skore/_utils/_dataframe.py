from typing import TYPE_CHECKING, Any, TypeAlias, cast

import narwhals as nw
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike

from skore._externals._sklearn_compat import check_array

if TYPE_CHECKING:
    import polars as pl

    UserDataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
    UserSeries: TypeAlias = pd.Series | pl.Series
else:
    UserDataFrame: TypeAlias = pd.DataFrame
    UserSeries: TypeAlias = pd.Series

UserTarget: TypeAlias = UserSeries | UserDataFrame


def _ensure_string_column_names(df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
    if all(isinstance(col, str) for col in df.columns):
        return df
    return df.rename({col: str(col) for col in df.columns})


def _normalize_X_as_dataframe(X: ArrayLike) -> UserDataFrame:
    """Normalize feature data as a DataFrame with string column names."""
    if sp.issparse(X):
        raise NotImplementedError(
            "Data analysis via skrub is currently not supported for sparse matrices. "
            "Please use dense data."
        )

    if not nw.dependencies.is_into_dataframe(X):
        X = check_array(X, accept_sparse=False, ensure_2d=True, ensure_all_finite=False)
        X = cast(np.ndarray, X)
        columns = [f"Feature {i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)

    return _ensure_string_column_names(nw.from_native(X)).to_native()


def _normalize_y_as_dataframe(y: ArrayLike) -> UserDataFrame:
    """Normalize target data as a DataFrame with predictable column names."""
    if sp.issparse(y):
        raise NotImplementedError(
            "Data analysis via skrub is currently not supported for sparse matrices. "
            "Please use dense data."
        )

    if nw.dependencies.is_into_series(y):
        if nw.dependencies.is_polars_series(y):
            series = nw.from_native(y, series_only=True)
            name = series.name if series.name else "Target"
            if not series.name:
                series = series.rename(name)
            return series.to_frame().to_native()

        y_series = cast(pd.Series, y)
        name = y_series.name if y_series.name is not None else "Target"
        return y_series.to_frame(name=name)

    if nw.dependencies.is_into_dataframe(y):
        if nw.dependencies.is_polars_dataframe(y):
            return y

        y_df = cast(pd.DataFrame, y)
        if all(isinstance(col, str) for col in y_df.columns):
            return y_df

        y_df = y_df.copy(deep=False)
        if y_df.shape[1] == 1 and list(y_df.columns) == [0]:
            y_df.columns = ["Target"]
        else:
            y_df.columns = [str(col) for col in y_df.columns]
        return y_df

    y = np.asarray(y)

    columns = ["Target"] if y.ndim == 1 else [f"Target {i}" for i in range(y.shape[1])]
    return pd.DataFrame(y, columns=columns)
