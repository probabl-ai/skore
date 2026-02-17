from collections.abc import Callable
from typing import Any, cast

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _num_features


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
        feature_names = estimator.feature_names_in_
        # skrub is not implementing rigorously the scikit-learn API and returns a
        # list and so we need to check if we have an array-like before to make any
        # conversion
        if hasattr(feature_names, "tolist"):
            return cast(list[str], cast(Any, feature_names).tolist())
        return cast(list[str], feature_names)

    elif (
        transformer is not None
        and hasattr(transformer, "get_feature_names_out")
        and _function_call_succeeds(cast(Callable, transformer.get_feature_names_out))
    ):
        return cast(Callable, transformer.get_feature_names_out)().tolist()
    elif X is not None:
        if hasattr(X, "columns"):
            return X.columns.tolist()  # type: ignore[assignment]
        else:
            return [f"Feature #{i}" for i in range(_num_features(X))]

    if n_features is None:
        raise ValueError(
            "Feature names cannot be inferred from the estimator or transformer. "
            "At least one of X or n_features must be provided to infer feature names."
        )
    return [f"Feature #{i}" for i in range(n_features)]
