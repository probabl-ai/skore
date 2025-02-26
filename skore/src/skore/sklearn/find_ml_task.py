"""A helper to guess the machine-learn task being performed."""

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn.types import MLTask


def _column_is_classification(y) -> bool:
    """Check whether ``y`` is a sequence.

    We define a sequence as a 1-d array of sequential integer values,
    where the first value is 0.
    """
    y_unique = np.unique(y)
    if np.any(y_unique == 0):
        sequential = np.arange(y_unique[0], y_unique[-1] + 1)
        return np.array_equal(y_unique, sequential)
    return False


def _is_classification(y) -> bool:
    """Determine if `y` is a target for a classification task.

    If `y` contains integers, sklearn's `type_of_target` considers the task
    to be multiclass classification. This might not be the case, so we add the
    constraints that `y` must contain sequential values and contain 0.

    If `y` is a 2-d array, we check each column independently; if at least one
    column does not fit the constraints, then we consider the task to be regression.

    Note that this can cause false positives, e.g. a classification task with classes
    0, 1, 2 where `y` contains no examples of class 1 would be falsely considered
    "regression".
    Similarly, if `y` is a 2-d array where some column contains 0 and 2 but not 1,
    the whole array would be considered "regression".

    If `y` does not contain numbers (e.g. strings) then this function returns True.
    """
    try:
        y = check_array(y, dtype="numeric", ensure_2d=False)
    except ValueError:
        # The conversion in `check_array` failed meaning
        # that we are in a classification case with non-numeric
        # data type.
        return True

    if y.ndim == 1:
        return _column_is_classification(y)

    # Iterate on columns of y (check_array ensures that y is at most 2-d)
    return all(_column_is_classification(column) for column in y.T)


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

    # Discrete sequential values, containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1], [2, 2], [1, 0]]))
    'multioutput-multiclass-classification'

    # Discrete values, not sequential, in a 2d array
    >>> _find_ml_task(numpy.array([[1, 5], [5, 9]]))
    'multioutput-regression'

    # 2 columns, one of them not containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1], [1, 1]]))
    'multioutput-regression'

    # Discrete values, not sequential, containing 0, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 1, 5, 9], [1, 0, 1, 1]]))
    'multioutput-regression'

    # 2 columns, one of them not sequential, in a 2d array
    >>> _find_ml_task(numpy.array([[0, 0], [2, 2], [1, 0]]))
    'multioutput-regression'
    """
    if estimator is not None:
        # checking the estimator is more robust and faster than checking the type of
        # target.
        if is_clusterer(estimator):
            return "clustering"
        if is_regressor(estimator):
            if y is None:
                return "regression"
            if np.ndim(y) == 1 or np.shape(y)[1] == 1:
                return "regression"
            return "multioutput-regression"
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
