"""CrossValidationReporter class.

This class implements a wrapper over scikit-learn's
`cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_
function in order to enrich it with more information and enable more analysis.
"""

from functools import cached_property

from .cross_validation_helpers import (
    _add_scorers,
    _get_scorers_to_add,
    _strip_cv_results_scores,
)
from .plot import plot_cross_validation


class CrossValidationReporter:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This class wraps scikit-learn's :func:`sklearn.model_selection.cross_validate`
    function, to provide more context and facilitate the analysis.
    As such, the arguments are the same as the
    :func:`sklearn.model_selection.cross_validate` function.

    For a user guide and in-depth example, see :ref:`example_cross_validate` and
    :ref:`example_track_cv`.

    More precisely, upon initialization, this class does the following:

    *   Detect the ML task being performed, based on the estimator and data

    *   Based on the ML task, add appropriate metrics to compute during
        cross-validation

    *   Perform the cross-validation itself

    *   Clean the cross-validation results so that the output of the function is as
        close as possible to scikit-learn's

    Parameters
    ----------
    estimator : estimator object implementing ‘fit’
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

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

    Attributes
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`sklearn.model_selection.cross_validate` function.

    X : array-like
        The data that was fitted.

    y : array-like or None
        The target variable, or None if not provided.

    plot : plotly.graph_objects.Figure
        A plot of the cross-validation results.

    Examples
    --------
    >>> def prepare_cv():
    ...     from sklearn import datasets, linear_model
    ...     diabetes = datasets.load_diabetes()
    ...     X = diabetes.data[:150]
    ...     y = diabetes.target[:150]
    ...     lasso = linear_model.Lasso()
    ...     return lasso, X, y

    >>> lasso, X, y = prepare_cv()  # doctest: +SKIP

    >>> reporter = CrossValidationReporter(lasso, X, y, cv=3)  # doctest: +SKIP
    CrossValidationReporter(...)
    """

    def __init__(self, *args, **kwargs):
        import sklearn.model_selection

        # Recover specific arguments
        self.estimator = args[0] if len(args) >= 1 else kwargs.get("estimator")
        self.X = args[1] if len(args) >= 2 else kwargs.get("X")
        self.y = args[2] if len(args) == 3 else kwargs.get("y")
        self.cv = kwargs.get("cv")

        self.scorers = kwargs.pop("scoring", None)
        return_estimator = kwargs.pop("return_estimator", None)
        return_indices = kwargs.pop("return_indices", None)

        # Extend scorers with other relevant scorers
        scorers_to_add = _get_scorers_to_add(self.estimator, self.y)
        self._scorers, added_scorers = _add_scorers(self.scorers, scorers_to_add)

        self._cv_results = sklearn.model_selection.cross_validate(
            *args,
            **kwargs,
            scoring=self._scorers,
            return_estimator=True,
            return_indices=True,
        )

        self._cv_results["fit_time_per_data_point"] = [
            time / nb_points
            for time, nb_points in zip(
                self._cv_results["fit_time"],
                map(len, self._cv_results["indices"]["train"]),
            )
        ]

        self._cv_results["score_time_per_data_point"] = [
            time / nb_points
            for time, nb_points in zip(
                self._cv_results["score_time"],
                map(len, self._cv_results["indices"]["test"]),
            )
        ]

        # Remove information related to our scorers, so that our return value is
        # the same as sklearn's
        self.cv_results = _strip_cv_results_scores(
            cv_results=self._cv_results,
            added_scorers=added_scorers,
            return_estimator=return_estimator,
            return_indices=return_indices,
        )

        # Add explicit metric to result (rather than just "test_score")
        if isinstance(self.scorers, str):
            if kwargs.get("return_train_score") is not None:
                self.cv_results[f"train_{self.scorers}"] = self.cv_results[
                    "train_score"
                ]
            self.cv_results[f"test_{self.scorers}"] = self.cv_results["test_score"]

    def __repr__(self):
        """Repr method."""
        return f"{self.__class__.__name__}(...)"

    @cached_property
    def plot(self):
        """Plot of the cross-validation results."""
        return plot_cross_validation(self._cv_results)
