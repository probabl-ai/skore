"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

import re
from io import BytesIO

import joblib
import pytest


@pytest.fixture(params=["report_estimator_reports", "report_cv_reports"])
def report(request):
    return request.getfixturevalue(request.param)


def test_help(capsys, report):
    """Check the help menu works."""
    report.help()

    captured = capsys.readouterr()
    assert "Tools to compare estimators" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_repr(report):
    """Check the `__repr__` works."""

    assert "ComparisonReport" in repr(report)


def test_metrics_repr(report):
    """Check the repr method of `report.metrics`."""
    repr_str = repr(report.metrics)
    assert "skore.ComparisonReport.metrics" in repr_str
    assert "help()" in repr_str


def test_pickle(tmp_path, report):
    """Check that we can pickle a comparison report."""
    with BytesIO() as stream:
        joblib.dump(report, stream)
        joblib.load(stream)


def test_cross_validation_report_cleaned_up(report):
    """
    When a CrossValidationReport is passed to a ComparisonReport, and computations are
    done on the ComparisonReport, the CrossValidationReport should remain pickle-able.

    Non-regression test for bug found in:
    https://github.com/probabl-ai/skore/pull/1512
    """
    report.metrics.report_metrics()

    with BytesIO() as stream:
        joblib.dump(report.reports_[0], stream)


def test_metrics_help(capsys, report):
    """Check that the help method writes to the console."""
    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out
