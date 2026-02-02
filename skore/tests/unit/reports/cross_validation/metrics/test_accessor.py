from skore import CrossValidationReport


def test_metrics_help(capsys, forest_binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_metrics_repr(forest_binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    repr_str = repr(report.metrics)
    assert "skore.CrossValidationReport.metrics" in repr_str
    assert "help()" in repr_str


def test_coefficients_help(capsys, linear_regression_data):
    """Check that the help method writes to the console"""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X=X, y=y)
    report.inspection.help()
    captured = capsys.readouterr()

    assert "Available model inspection methods" in captured.out
    assert "coefficients" in captured.out

    report.inspection.coefficients().help()
    captured = capsys.readouterr()

    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out


def test_coefficients_repr(linear_regression_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X=X, y=y)

    repr_str = repr(report.inspection)
    assert "skore.CrossValidationReport.inspection" in repr_str
    assert "help()" in repr_str
