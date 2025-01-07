"""Helpers for enhancing the cross-validation manipulation."""

from typing import Any

from sklearn import metrics

from skore.sklearn.find_ml_task import _find_ml_task


def _get_scorers_to_add(estimator, y) -> dict[str, Any]:
    """Get a list of scorers based on ``estimator`` and ``y``.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    scorers_to_add : dict[str, str]
        A list of scorers
    """
    ml_task = _find_ml_task(y, estimator)

    # Add scorers based on the ML task
    if ml_task == "regression":
        return {
            "r2": "r2",
            "root_mean_squared_error": metrics.make_scorer(
                metrics.root_mean_squared_error, response_method="predict"
            ),
        }
    if ml_task == "binary-classification":
        return {
            "roc_auc": "roc_auc",
            "brier_score_loss": metrics.make_scorer(
                metrics.brier_score_loss, response_method="predict_proba"
            ),
            "recall": "recall",
            "precision": "precision",
        }
    if ml_task == "multiclass-classification":
        if hasattr(estimator, "predict_proba"):
            return {
                "recall_weighted": "recall_weighted",
                "precision_weighted": "precision_weighted",
                "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
                "log_loss": metrics.make_scorer(
                    metrics.log_loss, response_method="predict_proba"
                ),
            }
        return {
            "recall_weighted": "recall_weighted",
            "precision_weighted": "precision_weighted",
        }
    return {}


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
    scorers_to_add : dict[str, str]
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
        # User-defined metrics have priority
        new_scorers = scorers_to_add | scorers
        added_scorers = set(scorers_to_add) - set(scorers)
    elif callable(scorers):
        from sklearn.metrics import check_scoring
        from sklearn.metrics._scorer import _MultimetricScorer

        internal_scorer = _MultimetricScorer(
            scorers={
                name: check_scoring(estimator=None, scoring=scoring)
                if isinstance(scoring, str)
                else scoring
                for name, scoring in scorers_to_add.items()
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
