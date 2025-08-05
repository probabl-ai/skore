import numpy as np
import pytest
from skore import CrossValidationReport


def test_cross_validation_report_metrics_help(capsys, forest_binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv_splitter=2)

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


