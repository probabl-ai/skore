from skore import EstimatorReport


def test_feature_importance_help(capsys, linear_regression_with_test):
    """Check that the help method writes to the console."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report.inspection.help()
    captured = capsys.readouterr()
    assert "Inspection accessor" in captured.out
    assert "coefficients" in captured.out

    report.inspection.coefficients().help()
    captured = capsys.readouterr()
    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out
