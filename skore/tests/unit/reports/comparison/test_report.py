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
    configuration,
)
from skore._sklearn._diagnostic import DiagnosticDisplay


@pytest.fixture(
    params=[
        "comparison_estimator_reports_binary_classification",
        "comparison_cross_validation_reports_binary_classification",
    ]
)
def report(request):
    return request.getfixturevalue(request.param)


def test_diagnose_collects_component_issues(report, monkeypatch):
    """Check that issues from all component reports are collected."""
    report_names = list(report.reports_)
    per_report_issues = [
        {
            f"SKD{i:03d}": {
                "title": f"Mock issue {i}",
                "docs_anchor": f"skd{i:03d}-mock",
                "explanation": f"Issue {i} detected.",
            }
        }
        for i, _ in enumerate(report_names, start=1)
    ]
    for sub_report, issues in zip(
        report.reports_.values(), per_report_issues, strict=True
    ):
        monkeypatch.setattr(
            sub_report,
            "_run_checks",
            lambda iss=issues: (iss, set(iss)),
        )
        if hasattr(sub_report, "_issues_cache"):
            delattr(sub_report, "_issues_cache")
    if hasattr(report, "_issues_cache"):
        delattr(report, "_issues_cache")

    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    for name, issues in zip(report_names, per_report_issues, strict=True):
        for code in issues:
            assert code in results.issues
            assert f"[{name}]" in results.issues[code]["explanation"]


def test_diagnose_uses_component_cache(report, monkeypatch):
    """Check that check results are cached and reused."""
    sub_report = next(iter(report.reports_.values()))
    calls = 0
    original = sub_report._run_checks

    def wrapped():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(sub_report, "_run_checks", wrapped)

    report.diagnose()
    report.diagnose()

    assert calls == 1


def test_diagnose_result_has_repr(report):
    """Check the diagnostic result has a repr."""
    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    assert "Diagnostic:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle


def test_diagnose_ignore(report, monkeypatch):
    """Check that checks are ignored when ignore is passed."""
    mock_issues = {
        "SKD001": {
            "title": "Mock overfitting",
            "docs_anchor": "skd001-overfitting",
            "explanation": "Mock overfitting detected.",
        }
    }
    for sub_report in report.reports_.values():
        monkeypatch.setattr(
            sub_report,
            "_run_checks",
            lambda: (mock_issues, {"SKD001", "SKD002"}),
        )
        if hasattr(sub_report, "_issues_cache"):
            delattr(sub_report, "_issues_cache")
    if hasattr(report, "_issues_cache"):
        delattr(report, "_issues_cache")
    results = report.diagnose(ignore=["SKD001"])
    assert "SKD001" not in results.issues


def test_diagnose_uses_global_ignore(report, monkeypatch):
    """Check that checks are ignored when global ignore is set."""
    mock_issues = {
        "SKD001": {
            "title": "Mock overfitting",
            "docs_anchor": "skd001-overfitting",
            "explanation": "Mock overfitting detected.",
        }
    }
    for sub_report in report.reports_.values():
        monkeypatch.setattr(
            sub_report,
            "_run_checks",
            lambda: (mock_issues, {"SKD001", "SKD002"}),
        )
        if hasattr(sub_report, "_issues_cache"):
            delattr(sub_report, "_issues_cache")
    if hasattr(report, "_issues_cache"):
        delattr(report, "_issues_cache")
    assert "SKD001" in report.diagnose().issues
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in report.diagnose().issues


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
