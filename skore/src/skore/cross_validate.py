"""cross_validate function.

This function implements a wrapper over scikit-learn's [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
function in order to enrich it with more information and enable more analysis.
"""

from skore.item.cross_validate_item import CrossValidateItem


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

    cv_results = sklearn.model_selection.cross_validate(*args, **kwargs)
    return CrossValidateItem.factory(cv_results)
