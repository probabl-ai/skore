"""CrossValidationReporter class.

This class implements a wrapper over scikit-learn's
`cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_
function in order to enrich it with more information and enable more analysis.
"""

from skore.sklearn.find_ml_task import _find_ml_task


def _get_scorers_to_add(estimator, y) -> list[str]:
    """Get a list of scorers based on ``estimator`` and ``y``.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    scorers_to_add : list[str]
        A list of scorers
    """
    ml_task = _find_ml_task(y, estimator)

    # Add scorers based on the ML task
    if ml_task == "regression":
        return ["r2", "neg_root_mean_squared_error"]
    if ml_task == "binary-classification":
        return ["roc_auc", "neg_brier_score", "recall", "precision"]
    if ml_task == "multiclass-classification":
        if hasattr(estimator, "predict_proba"):
            return [
                "recall_weighted",
                "precision_weighted",
                "roc_auc_ovr_weighted",
                "neg_log_loss",
            ]
        return ["recall_weighted", "precision_weighted"]
    return []


def _add_scorers(scorers, scorers_to_add):
    """Expand `scorers` with more scorers.

    The type of the resulting scorers object is dependent on the type of the input
    scorers:
    - If `scorers` is a dict, then extra scorers are added to the dict;
    - If `scorers` is a string or None, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a list or tuple, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a callable, then a new callable is created that
    returns a dict with the user-defined score as well as the scorers to add.
    In case the user-defined dict contains a metric with a name conflicting with the
    metrics we add, the user-defined metric always wins.

    Parameters
    ----------
    scorers : any type that is accepted by scikit-learn's cross_validate
        The scorer(s) to expand.
    scorers_to_add : list[str]
        The scorers to be added.

    Returns
    -------
    new_scorers : dict or callable
        The scorers after adding `scorers_to_add`.
    added_scorers : Iterable[str]
        The scorers that were actually added (i.e. the ones that were not already
        in `scorers`).
    """
    if scorers is None or isinstance(scorers, str):
        new_scorers, added_scorers = _add_scorers({"score": scorers}, scorers_to_add)
    elif isinstance(scorers, (list, tuple)):
        new_scorers, added_scorers = _add_scorers(
            {s: s for s in scorers}, scorers_to_add
        )
    elif isinstance(scorers, dict):
        new_scorers = {s: s for s in scorers_to_add} | scorers
        added_scorers = set(scorers_to_add) - set(scorers)
    elif callable(scorers):
        from sklearn.metrics import check_scoring
        from sklearn.metrics._scorer import _MultimetricScorer

        internal_scorer = _MultimetricScorer(
            scorers={
                s: check_scoring(estimator=None, scoring=s) for s in scorers_to_add
            }
        )

        def new_scorer(estimator, X, y) -> dict:
            scores = scorers(estimator, X, y)
            if isinstance(scores, dict):
                return internal_scorer(estimator, X, y) | scores
            return internal_scorer(estimator, X, y) | {"score": scores}

        new_scorers = new_scorer

        # In this specific case, we can't know if there is overlap between the
        # user-defined scores and ours, so we take the least risky option
        # which is to say we added nothing; that way, we won't remove anything
        # after cross-validation is computed
        added_scorers = []

    return new_scorers, added_scorers


def _strip_cv_results_scores(
    cv_results: dict,
    added_scorers: list[str],
    return_estimator: bool,
    return_indices: bool,
) -> dict:
    """Remove information about `added_scorers` in `cv_results`.

    Parameters
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's cross_validate function.
    added_scorers : list[str]
        A list of scorers in `cv_results` which should be removed.
    return_estimator : bool
        Whether to keep the "estimator" key or not.
    return_indices : bool
        Whether to keep the "indices" key or not.

    Returns
    -------
    dict
        A new cv_results dict, with the specified information removed.
    """
    _cv_results = cv_results.copy()

    if return_estimator is not True:
        del _cv_results["estimator"]

    if return_indices is not True:
        del _cv_results["indices"]

    # Takes care both of train and test scores
    _cv_results = {
        k: v
        for k, v in _cv_results.items()
        if not any(added_scorer in k for added_scorer in added_scorers)
    }

    return _cv_results


class CrossValidationReporter:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This class wraps scikit-learn's :func:`sklearn.model_selection.cross_validate`
    function, to provide more context and facilitate the analysis.
    As such, the arguments are the same as the
    :func:`sklearn.model_selection.cross_validate` function.

    For a user guide and in-depth example, see :ref:`example_cross_validate` and
    :ref:`example_track_cv`.

    More precisely, this class does the following:

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
