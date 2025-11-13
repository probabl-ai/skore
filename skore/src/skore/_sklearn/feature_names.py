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
    elif (
        transformer is not None
        and hasattr(transformer, "get_feature_names_out")
        and _function_call_succeeds(transformer.get_feature_names_out)
    ):
        return transformer.get_feature_names_out().tolist()
    elif X is not None:
        if hasattr(X, "columns"):
            return X.columns.tolist()
        else:
            return [f"Feature #{i}" for i in range(X.shape[1])]

    if n_features is None:
        raise ValueError(
            "Feature names cannot be inferred from the estimator or transformer. "
            "At least one of X or n_features must be provided to infer feature names."
        )
    return [f"Feature #{i}" for i in range(n_features)]
