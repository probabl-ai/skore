from collections.abc import Callable

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator


def _function_call_succeeds(func: Callable) -> bool:
    try:
        func()
        return True
    except AttributeError:
        return False


def _get_feature_names(
    estimator: BaseEstimator,
    *,
    transformer: BaseEstimator | None = None,
    X: ArrayLike | pd.DataFrame | None = None,
    n_features: int | None = None,
) -> list[str]:
    """Get the names of an estimator's input features.

    The estimator may or may not be inside a sklearn.Pipeline.
    """
    if hasattr(estimator, "feature_names_in_"):
        return estimator.feature_names_in_.tolist()
    elif transformer is not None and _function_call_succeeds(
        transformer.get_feature_names_out
    ):
        # It can happen that `transformer` does have `get_feature_names_out`, but
        # calling it fails because an underlying estimator does not have that method.
        return transformer.get_feature_names_out().tolist()
    elif X is not None and hasattr(X, "columns"):
        return X.columns.tolist()
    return [f"Feature #{i}" for i in range(n_features)]
