"""A helper to guess the machine-learn task being performed."""

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import type_of_target

from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn.types import MLTask


def _find_ml_task(y, estimator=None) -> MLTask:
    """Guess the ML task being addressed based on a target array and an estimator.

    Parameters
    ----------
    y : numpy.ndarray
        A target vector.
    estimator : sklearn.base.BaseEstimator, optional
        An estimator, used mainly if fitted.

    Returns
    -------
    MLTask
        The guess of the kind of ML task being performed.
    """
    if estimator is not None:
        # checking the estimator is more robust and faster than checking the type of
        # target.
        if is_clusterer(estimator):
            return "clustering"
        if is_regressor(estimator):
            return "regression"
        if is_classifier(estimator):
            if hasattr(estimator, "classes_"):  # fitted estimator
                if (
                    isinstance(estimator.classes_, np.ndarray)
                    and estimator.classes_.ndim == 1
                ):
                    if estimator.classes_.size == 2:
                        return "binary-classification"
                    if estimator.classes_.size > 2:
                        return "multiclass-classification"
            else:  # fallback on the target
                target_type = type_of_target(y)
                if target_type == "binary":
                    return "binary-classification"
                if target_type == "multiclass":
                    return "multiclass-classification"
            return "unsupported"
        return "unsupported"
    else:
        if y is None:
            # NOTE: The task might not be clustering
            return "clustering"

        target_type = type_of_target(y)

        if target_type == "continuous":
            return "regression"
        if target_type == "binary":
            return "binary-classification"
        if target_type == "multiclass":
            return "multiclass-classification"
        return "unsupported"
