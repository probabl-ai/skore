"""
Tests of ComparisonReport which work regardless whether it holds EstimatorReports or
CrossValidationReports.
"""

from io import BytesIO

import joblib
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils._testing import _convert_container

from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
)


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
    sub_report = next(iter(report.reports_.values()))

    with BytesIO() as stream:
        joblib.dump(sub_report, stream)


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


@pytest.mark.parametrize(
    "container_types", [("dataframe", "series"), ("array", "array")]
)
def test_create_estimator_report_from_estimator_reports(
    container_types, binary_classification_data
):
    """Test creating an estimator report from a comparison report with
    EstimatorReports."""
    X, y = binary_classification_data
    X = _convert_container(X, container_types[0])
    y = _convert_container(y, container_types[1])
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    estimators = {
        "estimator_1": LinearSVC(random_state=42),
        "estimator_2": LogisticRegression(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_experiment, y_experiment, test_size=0.2, random_state=42, shuffle=False
    )
    comparison_report = ComparisonReport(
        {
            name: EstimatorReport(
                est, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            for name, est in estimators.items()
        }
    )

    est_report_w_test = comparison_report.create_estimator_report(
        report_key="estimator_2", X_test=X_heldout, y_test=y_heldout
    )

    assert isinstance(est_report_w_test, EstimatorReport)
    assert joblib.hash(est_report_w_test.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report_w_test.y_train) == joblib.hash(y_experiment)
    assert joblib.hash(est_report_w_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_w_test.y_test) == joblib.hash(y_heldout)


@pytest.mark.parametrize(
    "container_types", [("dataframe", "series"), ("array", "array")]
)
def test_create_estimator_report_from_cross_validation_reports(
    container_types, binary_classification_data
):
    """Test creating an estimator report from a comparison report with
    CrossValidationReports."""
    X, y = binary_classification_data
    X = _convert_container(X, container_types[0])
    y = _convert_container(y, container_types[1])
    X_experiment, X_heldout, y_experiment, y_heldout = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    estimators = {
        "estimator_1": LinearSVC(random_state=42),
        "estimator_2": LogisticRegression(random_state=42),
    }

    reports = {
        name: CrossValidationReport(est, X=X_experiment, y=y_experiment, splitter=2)
        for name, est in estimators.items()
    }

    comparison_report = ComparisonReport(reports)

    est_report_w_test = comparison_report.create_estimator_report(
        report_key="estimator_2", X_test=X_heldout, y_test=y_heldout
    )

    assert isinstance(est_report_w_test, EstimatorReport)
    assert joblib.hash(est_report_w_test.X_train) == joblib.hash(X_experiment)
    assert joblib.hash(est_report_w_test.y_train) == joblib.hash(y_experiment)
    assert joblib.hash(est_report_w_test.X_test) == joblib.hash(X_heldout)
    assert joblib.hash(est_report_w_test.y_test) == joblib.hash(y_heldout)


def test_create_estimator_report_invalid_name(
    comparison_estimator_reports_binary_classification,
):
    """Test that an error is raised when an invalid estimator name is provided."""
    comparison_report = comparison_estimator_reports_binary_classification

    err_msg = "Estimator with key InvalidEstimator not found in the comparison report."
    with pytest.raises(ValueError, match=err_msg):
        comparison_report.create_estimator_report(
            report_key="InvalidEstimator", X_test=[0], y_test=None
        )


@pytest.mark.parametrize(
    "comparison_fixture",
    [
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
        "comparison_estimator_reports_multiclass_classification",
        "comparison_cross_validation_reports_multiclass_classification",
        "comparison_estimator_reports_regression",
        "comparison_cross_validation_reports_regression",
        "comparison_estimator_reports_multioutput_regression",
        "comparison_cross_validation_reports_multioutput_regression",
    ],
)
def test_report_repr_html(comparison_fixture, request):
    report = request.getfixturevalue(comparison_fixture)
    sub_report = next(iter(report.reports_.values()))
    expected_estimator_name = sub_report.estimator_.__class__.__name__
    html_out = report._repr_html_()
    assert "skore-comparison-report-" in html_out
    assert "Model comparison" in html_out
    assert expected_estimator_name in html_out
    assert "skoreInitComparisonReport" in html_out
    assert "report-hint-note" in html_out
    assert "docs.skore.probabl.ai" in html_out
    assert "report-tabset" in html_out
    assert "ComparisonReport.metrics" in html_out
    assert "skore-comparison-report-select" in html_out
