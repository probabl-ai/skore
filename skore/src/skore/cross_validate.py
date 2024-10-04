"""cross_validate function.

This function implements a wrapper over scikit-learn's [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
function in order to enrich it with more information and enable more analysis.
"""

from typing import Literal

from skore.item.cross_validate_item import CrossValidateItem


def _find_ml_task(
    estimator,
    y,
) -> Literal["classification", "regression", "clustering", "unknown"]:
    import sklearn.utils.multiclass
    from sklearn.base import is_classifier, is_regressor

    if y is None:
        # FIXME: The task might not be clustering
        return "clustering"

    if is_classifier(estimator):
        return "classification"

    if is_regressor(estimator):
        return "regression"

    type_of_target = sklearn.utils.multiclass.type_of_target(y)

    if type_of_target == "unknown":
        return "unknown"

    if "continuous" in type_of_target:
        return "regression"

    return "classification"


def _expand_scorers(scorers, scorers_to_add: list[str]):
    """Expand `scorers` with `scorers_to_add`.

    Parameters
    ----------
    scorers : any type that is accepted by scikit-learn's cross_validate
        The scorer(s) to expand.
    scorers_to_add : list[str]
        The scorers to be added

    Returns
    -------
    new_scorers : dict[str, str | None]
        The scorers after adding `scorers_to_add`.
    added_scorers : list[str]
        The scorers that were actually added (i.e. the ones that were not already
        in `scorers`).
    """
    added_scorers = []

    if scorers is None:
        new_scorers = {"score": None}
        for s in scorers_to_add:
            new_scorers[s] = s
            added_scorers.append(s)
    elif isinstance(scorers, str):
        new_scorers = {"score": scorers}
        for s in scorers_to_add:
            if s == scorers:
                continue
            new_scorers[s] = s
            added_scorers.append(s)
    elif isinstance(scorers, dict):
        new_scorers = scorers.copy()
        for s in scorers_to_add:
            if s in scorers:
                continue
            new_scorers[s] = s
            added_scorers.append(s)
    elif isinstance(scorers, list):
        new_scorers = scorers.copy()
        for s in scorers_to_add:
            if s in scorers:
                continue
            new_scorers.append(s)
            added_scorers.append(s)
    elif isinstance(scorers, tuple):
        scorers = list(scorers)
        new_scorers, added_scorers = _expand_scorers(scorers, scorers_to_add)

    return new_scorers, added_scorers
def cross_validate(*args, **kwargs) -> CrossValidateItem:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This function wraps scikit-learn's [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
    function, to provide more context
    and facilitate the analysis.
    As such, the arguments are the same as scikit-learn's cross_validate function.

    Parameters
    ----------
    The same parameters as scikit-learn's cross_validate function.

    Returns
    -------
    CrossValidateItem
        An object containing the cross-validation results, which can be readily
        inserted into a Project.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> cv_results
    CrossValidateItem(...)
    """
    import sklearn.model_selection

    # Extend scorers with other relevant scorers

    # Recover target
    estimator = args[0] if len(args) >= 1 else kwargs.get("estimator")
    y = args[2] if len(args) == 3 else kwargs.get("y")

    ml_task = _find_ml_task(estimator, y)

    # Add scorers based on the ML task
    if ml_task == "clustering":
        scorers_to_add = ["homogeneity_score", "silhouette_score", "rand_score"]
    elif ml_task == "regression":
        scorers_to_add = ["r2", "neg_mean_squared_error"]
    elif ml_task == "classification":
        scorers_to_add = ["roc_auc", "neg_brier_score", "recall", "precision"]
    elif ml_task == "unknown":
        scorers_to_add = []

    try:
        scorers = kwargs.pop("scoring")
    except KeyError:
        scorers = None
    new_scorers, added_scorers = _expand_scorers(scorers, scorers_to_add)

    # store test_score in skore associated with the following metadata to recognize the run:
    # estimator (name & parameters)
    # hash of y (if present)
    # to recognize df_train, the following set {nb_col, nb_rows, hash}. This is suboptimal, but good for an MVP.
    # the plot (1) is displayed in the notebook if the code is ran in notebook
    # the plot (1) is created in skore UI

    cv_results = sklearn.model_selection.cross_validate(
        *args, **kwargs, scoring=new_scorers
    )
