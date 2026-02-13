from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skrub import _dataframe as sbd

from skore._externals._sklearn_compat import check_array


def _normalize_X_as_dataframe(X: ArrayLike) -> pd.DataFrame:
    """Normalize feature data as a pandas DataFrame with string column names."""
    if not sbd.is_dataframe(X):
        X = check_array(X, accept_sparse=False, ensure_2d=True, ensure_all_finite=False)
        X = cast(np.ndarray, X)
        columns = [f"Feature {i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)

    X_df = cast(pd.DataFrame, X)
    if all(isinstance(col, str) for col in X_df.columns):
        return X_df

    X_df = X_df.copy(deep=False)
    X_df.columns = [str(col) for col in X_df.columns]
    return X_df


def _normalize_y_as_dataframe(y: ArrayLike) -> pd.DataFrame:
    """Normalize target data as a pandas DataFrame with predictable column names."""
    if isinstance(y, pd.Series):
        name = y.name if y.name is not None else "Target"
        return y.to_frame(name=name)

    if sbd.is_dataframe(y):
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
