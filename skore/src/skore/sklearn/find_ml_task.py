"""A helper to guess the machine-learn task being performed."""

from skore.sklearn.types import MLTask


def _find_ml_task(y, estimator=None) -> MLTask:
    """Guess the ML task being addressed based on a target array and an estimator.

    Parameters
    ----------
    y : numpy.ndarray
        A target vector.
    estimator : sklearn.base.BaseEstimator, optional
        An estimator.

    Returns
    -------
    MLTask
        The guess of the kind of ML task being performed.
    """
    import sklearn.utils.multiclass
    from sklearn.base import is_classifier, is_regressor

    if y is None:
        # NOTE: The task might not be clustering
        return "clustering"

    if is_regressor(estimator):
        return "regression"

    type_of_target = sklearn.utils.multiclass.type_of_target(y)

    if is_classifier(estimator):
        if type_of_target == "binary":
            return "binary-classification"

        if type_of_target == "multiclass":
            return "multiclass-classification"

    if type_of_target == "unknown":
        return "unknown"

    if "continuous" in type_of_target:
        return "regression"

    return "classification"
