"""cross_validate function.

This function implements a wrapper over scikit-learn's
`cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_
function in order to enrich it with more information and enable more analysis.
"""

import contextlib
from typing import Optional

from skore.item.cross_validation_item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
)
from skore.project import Project


def cross_validate(*args, project: Optional[Project] = None, **kwargs) -> dict:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This function wraps scikit-learn's :func:`sklearn.model_selection.cross_validate`
    function, to provide more context and facilitate the analysis.
    As such, the arguments are the same as the
    :func:`sklearn.model_selection.cross_validate` function.

    The dict returned by this function is a strict super-set of the one returned by
    :func:`sklearn.model_selection.cross_validate`.

    For a user guide and in-depth example, see :ref:`example_cross_validate` and
    :ref:`example_track_cv`.

    More precisely, this function does the following:

    *   Detect the ML task being performed, based on the estimator and data

    *   Based on the ML task, add appropriate metrics to compute during
        cross-validation

    *   Perform the cross-validation itself

    *   Save the result to ``project``, if available

    *   Clean the cross-validation results so that the output of the function is as
        close as possible to scikit-learn's

    *   Return the clean cross-validation results.

    Parameters
    ----------
    estimator : estimator object implementing ‘fit’
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    project : Project, default=None
        A project to save cross-validation data into. If None, no save is performed.

    groups : array-like of shape (n_samples,), default=None
        See :func:`sklearn.model_selection.cross_validate`.

    scoring : str, callable, list, tuple, or dict, default=None
        See :func:`sklearn.model_selection.cross_validate`.

    cv : int, cross-validation generator or an iterable, default=None
        See :func:`sklearn.model_selection.cross_validate`.

    n_jobs : int, default=None
        See :func:`sklearn.model_selection.cross_validate`.

    verbose : int, default=0
        See :func:`sklearn.model_selection.cross_validate`.

    params : dict, default=None
        See :func:`sklearn.model_selection.cross_validate`.

    pre_dispatch : int or str, default=’2*n_jobs’
        See :func:`sklearn.model_selection.cross_validate`.

    return_train_score : bool, default=False
        See :func:`sklearn.model_selection.cross_validate`.

    return_estimator : bool, default=False
        See :func:`sklearn.model_selection.cross_validate`.

    return_indices : bool, default=False
        See :func:`sklearn.model_selection.cross_validate`.

    error_score : 'raise' or numeric, default=np.nan
        See :func:`sklearn.model_selection.cross_validate`.

    Returns
    -------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`sklearn.model_selection.cross_validate` function.

    Examples
    --------
    >>> def prepare_cv():
    ...     from sklearn import datasets, linear_model
    ...     diabetes = datasets.load_diabetes()
    ...     X = diabetes.data[:150]
    ...     y = diabetes.target[:150]
    ...     lasso = linear_model.Lasso()
    ...     return lasso, X, y

    >>> project = skore.load("project.skore")  # doctest: +SKIP
    >>> lasso, X, y = prepare_cv()  # doctest: +SKIP
    >>> cross_validate(lasso, X, y, cv=3, project=project)  # doctest: +SKIP
    {'fit_time': array(...), 'score_time': array(...), 'test_score': array(...)}
    """
    import sklearn.model_selection

    # Recover specific arguments
    estimator = args[0] if len(args) >= 1 else kwargs.get("estimator")
    X = args[1] if len(args) >= 2 else kwargs.get("X")
    y = args[2] if len(args) == 3 else kwargs.get("y")

    try:
        scorers = kwargs.pop("scoring")
    except KeyError:
        scorers = None

    # Extend scorers with other relevant scorers
    scorers_to_add = _get_scorers_to_add(estimator, y)
    new_scorers, added_scorers = _add_scorers(scorers, scorers_to_add)

    cv_results = sklearn.model_selection.cross_validate(
        *args, **kwargs, scoring=new_scorers
    )

    cross_validation_item = CrossValidationItem.factory(cv_results, estimator, X, y)

    if project is not None:
        try:
            cv_results_history = project.get_item_versions("cross_validation")
        except KeyError:
            cv_results_history = []

        agg_cross_validation_item = CrossValidationAggregationItem.factory(
            cv_results_history + [cross_validation_item]
        )

        project.put_item("cross_validation_aggregated", agg_cross_validation_item)
        project.put_item("cross_validation", cross_validation_item)

    # If in a IPython context (e.g. Jupyter notebook), display the plot
    with contextlib.suppress(ImportError):
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.display import display

        if InteractiveShell.initialized():
            display(cross_validation_item.plot)

    # Remove information related to our scorers, so that our return value is
    # the same as sklearn's
    stripped_cv_results = _strip_cv_results_scores(cv_results, added_scorers)

    # Add explicit metric to result (rather than just "test_score")
    if isinstance(scorers, str):
        if kwargs.get("return_train_score") is not None:
            stripped_cv_results[f"train_{scorers}"] = stripped_cv_results["train_score"]
        stripped_cv_results[f"test_{scorers}"] = stripped_cv_results["test_score"]

    return stripped_cv_results
