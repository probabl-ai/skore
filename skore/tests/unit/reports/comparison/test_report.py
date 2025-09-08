"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

import re
from io import BytesIO

import joblib
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skore import ComparisonReport, CrossValidationReport, EstimatorReport


@pytest.fixture(
    params=[
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ]
)
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
    report.metrics.summarize()

    with BytesIO() as stream:
        joblib.dump(report.reports_[0], stream)


def test_metrics_help(capsys, report):
    """Check that the help method writes to the console."""
    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_feature_importance_help(capsys):
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    reports = {
        name: EstimatorReport(est, X_train=X, X_test=X, y_train=y, y_test=y)
        for name, est in estimators.items()
    }

    comparison_report = ComparisonReport(reports)

    comparison_report.feature_importance.help()
    captured = capsys.readouterr()

    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out

    comparison_report.feature_importance.coefficients().help()
    captured = capsys.readouterr()

    assert "frame" in captured.out
    assert "plot" in captured.out
    assert "set_style" in captured.out


@pytest.mark.parametrize("report", [EstimatorReport, CrossValidationReport])
def test_pos_label_mismatch(report):
    """Check that we raise an error when the positive labels are not the same."""
    X, y = make_classification(random_state=0)
    estimators = {"LinearSVC": LinearSVC(), "LogisticRegression": LogisticRegression()}

    if report is EstimatorReport:
        reports = {
            name: EstimatorReport(
                est, X_train=X, X_test=X, y_train=y, y_test=y, pos_label=i
            )
            for i, (name, est) in enumerate(estimators.items())
        }
    else:
        reports = {
            name: CrossValidationReport(est, X=X, y=y, pos_label=i)
            for i, (name, est) in enumerate(estimators.items())
        }

    err_msg = "Expected all estimators to have the same positive label."
    with pytest.raises(ValueError, match=err_msg):
        ComparisonReport(reports)
