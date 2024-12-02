from skore.sklearn.cross_validation_reporter import CrossValidationReporter


def prepare_cv():
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()
    return lasso, X, y


def test_cross_validate():
    """When the user doesn't pass `return_estimator=True` or `return_indices=True`,
    the user-facing results don't have the associated keys but the internal ones do."""
    lasso, X, y = prepare_cv()
    reporter = CrossValidationReporter(lasso, X, y, cv=3)

    assert "estimator" not in reporter.cv_results
    assert "indices" not in reporter.cv_results
    assert "estimator" in reporter._cv_results
    assert "indices" in reporter._cv_results


def test_cross_validate_return_estimator():
    """When the user passes `return_estimator=True` and `return_indices=True`,
    the user-facing results have the associated keys and the internal ones do."""
    lasso, X, y = prepare_cv()
    reporter = CrossValidationReporter(
        lasso, X, y, cv=3, return_estimator=True, return_indices=True
    )

    assert "estimator" in reporter.cv_results
    assert "indices" in reporter.cv_results
    assert "estimator" in reporter._cv_results
    assert "indices" in reporter._cv_results
