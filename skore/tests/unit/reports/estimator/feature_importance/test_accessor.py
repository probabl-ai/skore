from skore import EstimatorReport


def test_feature_importance_help(capsys, linear_regression_with_test):
    """Check that the help method writes to the console."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report.feature_importance.help()
    captured = capsys.readouterr()
    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out


def test_feature_importance_repr(linear_regression_with_test):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    repr_str = repr(report.feature_importance)
    assert "skore.EstimatorReport.feature_importance" in repr_str
    assert "help()" in repr_str
