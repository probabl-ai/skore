"""A helper to guess the machine-learn task being performed."""

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import type_of_target

from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn.types import MLTask


def _is_sequential(y) -> bool:
    """Check whether ``y`` is vector of sequential integer values."""
    y_values = np.sort(np.unique(y))
    sequential = np.arange(y_values[0], y_values[-1] + 1)
    return np.array_equal(y_values, sequential)


def _is_classification(y) -> bool:
    """Determine if `y` is a target for a classification task.

    If `y` contains integers, sklearn's `type_of_target` considers
    the task to be multiclass classification.
    This function makes the analysis finer.
    """
    y = y.flatten()
    return _is_sequential(y) and 0 in y


def _find_ml_task(y, estimator=None) -> MLTask:
    """Guess the ML task being addressed based on a target array and an estimator.

    This relies first on the estimator characteristics, and falls back on
    analyzing ``y``. Check the examples for some of the heuristics relied on.

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

    Examples
    --------
    >>> import numpy
    >>> from skore.sklearn.find_ml_task import _find_ml_task

    # Discrete values, not sequential
    >>> _find_ml_task(numpy.array([1, 5, 9]))
    'regression'

    # Discrete values, not sequential, containing 0
    >>> _find_ml_task(numpy.array([0, 1, 5, 9]))
    'regression'

    # Discrete sequential values, containing 0
    >>> _find_ml_task(numpy.array([0, 1, 2]))
    'multiclass-classification'

    # Discrete sequential values, not containing 0
    >>> _find_ml_task(numpy.array([1, 3, 2]))
    'regression'

    # 2 values, containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1], [1, 1]]))
    'multioutput-binary-classification'

    # Discrete sequential values, containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1, 2], [2, 1, 1]]))
    'multioutput-multiclass-classification'

    # Discrete values, not sequential, in a 2d array
    >>> _find_ml_task(numpy.array([[1, 5], [5, 9]]))
    'multioutput-regression'

    # Discrete values, not sequential, containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1, 5, 9], [1, 0, 1, 1]]))
    'multioutput-regression'

    # Discrete sequential values, not containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[1, 3, 2], [2, 1, 1]]))
    'multioutput-regression'
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
            else:
                # fallback on the target
                if y is None:
                    return "unknown"

    # fallback on the target
    if y is None:
        # NOTE: The task might not be clustering
        return "clustering"

    target_type = type_of_target(y)

    if target_type == "continuous":
        return "regression"
    if target_type == "continuous-multioutput":
        return "multioutput-regression"
    if target_type == "binary":
        return "binary-classification"
    if target_type == "multiclass":
        if _is_classification(y):
            return "multiclass-classification"
        return "regression"
    if target_type == "multiclass-multioutput":
        if _is_classification(y):
            return "multioutput-multiclass-classification"
        return "multioutput-regression"
    if target_type == "multilabel-indicator":
        if _is_classification(y):
            return "multioutput-binary-classification"
        return "multioutput-regression"
    return "unknown"
