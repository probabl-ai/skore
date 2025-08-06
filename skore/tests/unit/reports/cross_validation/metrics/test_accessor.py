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
