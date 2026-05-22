from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from skrub import tabular_pipeline

from skore import Check, EstimatorReport, configuration, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import (
    ChecksSummaryDisplay,
    _get_issue_documentation_url,
)


@pytest.fixture(params=[LinearRegression(), tabular_pipeline(LinearRegression())])
def regression_report(request, regression_data):
    estimator = request.param
    X, y = regression_data
    return evaluate(
        estimator,
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
    )


def mock_issue(report, ignored_codes):
    return (
        {
            "SKD001": {
                "title": "Mock title",
                "docs_url": "skd001-overfitting",
                "explanation": "Mock overfitting detected.",
                "severity": "issue",
            }
        },
        {"SKD001"},
    )


class MockCheck(Check):
    code = "TST001"
    title = "Test issue"
    report_type = "estimator"
    docs_url = "tst001"

    def __init__(
        self, has_issue: bool = True, docs_url="tst001", report_type="estimator"
    ):
        self.has_issue = has_issue
        self.docs_url = docs_url
        self.report_type = report_type

    def check_function(self, report):
        return "Something was found." if self.has_issue else None


def test_skd001_detects_overfitting(regression_data):
    """Check that the overfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD001" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} default predictive metrics"
        in issues.loc["SKD001", "explanation"]
    )


def test_skd002_detects_underfitting(regression_data):
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


@pytest.mark.parametrize(
    "weights, code", [([0.9, 0.1], "SKD004"), ([0.9, 0.05, 0.05], "SKD005")]
)
def test_skd004_skd005_detects_high_class_imbalance(weights, code):
    """Check that the high class imbalance issue is detected."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    result = report.checks.summarize()
    assert code not in set(result.frame(severity="issue")["code"])

    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        weights=weights,
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    assert code in issues.index
    assert "Accuracy should not be used alone" in issues.loc[code, "explanation"]


def test_skd006_detects_coefficient_interpretation(regression_data):
    """Check that the coefficient interpretation tip is emitted."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(severity="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(severity="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


def test_skd007_mdi_bias_with_high_cardinality(regression_data):
    """SKD007 tip is emitted with continuous features and tree importances."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(n_estimators=5, random_state=0), X, y)
    tips = report.checks.summarize().frame(severity="tip").set_index("code")
    assert "SKD007" in tips.index
    assert (
        "High-cardinality features detected: 0, 1, 2 (and 1 more)"
        in tips.loc["SKD007", "explanation"]
    )


def test_skd007_not_emitted_for_binary_features():
    """SKD007 tip is absent when all features are low-cardinality."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 2, size=(20, 4)).astype(float)
    y = rng.standard_normal(20)
    report = evaluate(RandomForestRegressor(n_estimators=5, random_state=0), X, y)
    tips = report.checks.summarize().frame(severity="tip").set_index("code")
    assert "SKD007" not in tips.index


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), tabular_pipeline(LinearRegression())]
)
def test_skd008_correlated_features(estimator):
    """SKD008 issue is emitted when two features are near-perfectly correlated."""
    rng = np.random.RandomState(42)
    X = rng.standard_normal((20, 4))
    X[:, 1] = X[:, 0] + rng.standard_normal(20) * 1e-4
    y = rng.standard_normal(20)
    report = evaluate(
        estimator,
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
    )
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    assert "SKD008" in issues.index
    assert "1 pair(s) of features" in issues.loc["SKD008", "explanation"]


def test_skd008_not_emitted_for_independent_features(regression_data):
    """SKD008 issue is absent when features are independent."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    assert "SKD008" not in issues.index


def test_skd009_detects_worse_than_baseline(regression_data):
    """Check that the worse-than-baseline issue is detected on a dummy estimator."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    assert "SKD009" in issues.index
    assert (
        "not significantly better than a HistGradientBoosting baseline"
        in issues.loc["SKD009", "explanation"]
    )


def test_skd009_not_detected_on_strong_model(regression_data):
    """Check that SKD009 is not detected when the model beats HistGradientBoosting."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(severity="issue")["code"])
    assert "SKD009" not in codes


def test_skd010_detects_slower_than_baseline(regression_data):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(n_estimators=200, random_state=0), X, y)
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    assert "SKD010" in issues.index
    assert "slower than a fast linear baseline" in issues.loc["SKD010", "explanation"]


def test_skd010_not_detected_for_fast_model(regression_data):
    """Check that SKD010 does not fire when the model is not slower than baseline."""
    X, y = regression_data
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(severity="issue")["code"])
    assert "SKD010" not in codes


def test_ignore_checks(monkeypatch, regression_report):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert (
        regression_report.checks.summarize(ignore=["SKD001"])
        .frame(severity="issue")
        .empty
    )


def test_exception_when_train_data_missing(regression_train_test_split):
    """Check that an exception is raised when the train data is missing."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    for check in report._checks_registry:
        if check.code in ["SKD001", "SKD002", "SKD009", "SKD010"]:
            with pytest.raises(CheckNotApplicable):
                check.check_function(report)


def test_exception_when_baseline_report_creation_fails(regression_data, monkeypatch):
    """Check that an exception is raised when the baseline report creation fails."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    monkeypatch.setattr(
        EstimatorReport,
        "_fit_estimator",
        lambda self, **kwargs: RuntimeError("Test error"),
    )
    for check in report._checks_registry:
        if check.code in ["SKD002", "SKD009", "SKD010"]:
            with pytest.raises(CheckNotApplicable):
                check.check_function(report)


def test_no_issues(monkeypatch, regression_report):
    """Check that no issues are detected when checks pass."""
    monkeypatch.setattr(
        EstimatorReport,
        "_get_results",
        lambda report, ignored_codes: ({}, {"SKD001", "SKD002"}),
    )
    assert regression_report.checks.summarize().frame(severity="issue").empty


def test_checks_summary_repr(monkeypatch, regression_report):
    """Check that the checks summary has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    results = regression_report.checks.summarize()
    assert isinstance(results, ChecksSummaryDisplay)
    elements = [
        "Checks summary:",
        "Mock title.",
        "[SKD001]",
        "Mock overfitting detected",
    ]
    for element in elements:
        assert element in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/automated_checks.html#" in bundle["text/html"]


def test_global_ignore(monkeypatch, regression_report):
    """Check that checks are ignored when global ignore is set."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert "SKD001" in set(
        regression_report.checks.summarize().frame(severity="issue")["code"]
    )
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in set(
            regression_report.checks.summarize().frame(severity="issue")["code"]
        )


def test_documentation_url_points_to_existing_rst():
    """Check that the URL in _get_issue_documentation_url maps to a real RST file."""
    url = urlparse(_get_issue_documentation_url(mock_issue(None, set())[0]["SKD001"]))
    # url.path is e.g. "/dev/user_guide/automated_checks.html"
    # strip version prefix and convert .html -> .rst
    rst_rel_path = "/".join(url.path.split("/")[2:]).replace(".html", ".rst")
    rst_path = Path(__file__).parents[5] / "sphinx" / rst_rel_path
    assert rst_path.is_file()


def test_reuses_cached_results(monkeypatch, regression_report):
    """Check that check results are cached and reused."""
    calls = 0
    original_run = Check.check_function

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "check_function", counting_run)
    regression_report.checks.summarize()
    calls_after_first = calls
    regression_report.checks.summarize()
    assert calls == calls_after_first


def test_add_checks_runs_custom_check(regression_report):
    """Check that add_checks runs the custom check and includes its issue."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    issues = (
        regression_report.checks.summarize().frame(severity="issue").set_index("code")
    )
    assert "TST001" in issues.index
    assert issues.loc["TST001", "title"] == "Test issue"
    assert issues.loc["TST001", "documentation_url"].endswith("#tst001")
    assert issues.loc["TST001", "explanation"] == "Something was found."


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_report):
    """Check that add_checks does not re-run already cached built-in checks."""
    regression_report.checks.summarize()

    for check in regression_report._checks_registry:
        monkeypatch.setattr(
            check, "check_function", lambda report: pytest.fail("re-ran cached check")
        )

    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()


def test_add_checks_docs_url_full(regression_report):
    """Check that a full https docs_url is preserved as-is in frame()."""
    check = MockCheck(has_issue=True, docs_url="https://example.com/my-doc")
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame.query("code == 'TST001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_report):
    """Check that missing docs_url results in None in the frame."""
    check = MockCheck(has_issue=True, docs_url=None)
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame[frame["code"] == "TST001"]
    assert row["documentation_url"].isna().all()


def test_available_returns_code_dash_title(regression_report):
    """Check that available returns strings in 'code - title' format."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    available = regression_report.checks.available()
    assert "TST001 - Test issue" in available


def test_remove_checks_excludes_results(regression_report):
    """Check that remove excludes checks from results and available checks."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001" in set(regression_report.checks.summarize().frame()["code"])

    regression_report.checks.remove("TST001")
    assert "TST001" not in set(regression_report.checks.summarize().frame()["code"])
    assert "TST001 - Test issue" not in regression_report.checks.available()


def test_remove_clears_cache(regression_report):
    """Check that remove invalidates cached results for the removed check."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()
    assert "TST001" in regression_report._check_results_cache
    assert "TST001" in regression_report._applicable_codes

    regression_report.checks.remove("TST001")
    assert "TST001" not in regression_report._check_results_cache
    assert "TST001" not in regression_report._applicable_codes


def test_remove_is_case_insensitive(regression_report):
    """Check that remove matches check codes case-insensitively."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001 - Test issue" in regression_report.checks.available()
    regression_report.checks.remove("tst001")
    assert "TST001 - Test issue" not in regression_report.checks.available()


def test_check_invalid_report_type(regression_report):
    """Check that Check raises ValueError for unsupported report_type."""
    check = MockCheck(has_issue=False, report_type="invalid")
    with pytest.raises(ValueError, match="report_type should be one of"):
        regression_report.checks.add([check])


def test_check_invalid_protocol(regression_report):
    """Check that Check raises ValueError for unsupported protocol."""

    class InvalidCheck:
        code = "INVALID001"
        title = "Invalid issue"
        report_type = "estimator"
        docs_url = "invalid001"

    with pytest.raises(ValueError, match="does not implement the Check protocol."):
        regression_report.checks.add([InvalidCheck()])


def test_custom_metric(binary_classification_data):
    """Check that checks works with custom metrics in the report."""
    X, y = binary_classification_data
    report = evaluate(DummyClassifier(), X, y, pos_label=1)
    report.metrics.add("f1")
    issues = report.checks.summarize().frame(severity="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


class TipCheck(Check):
    code = "TST002"
    title = "Tip check"
    report_type = "estimator"
    docs_url = "tst_tip"
    severity = "tip"

    def check_function(self, report):
        return "Be careful about this."


def test_tip_goes_to_tips_not_issues(regression_report):
    """A check with severity='tip' is routed to tips, not issues."""
    regression_report.checks.add([TipCheck()])
    result = regression_report.checks.summarize()
    tips = result.frame(severity="tip").set_index("code")
    assert "TST002" in tips.index
    assert "TST002" not in set(result.frame(severity="issue")["code"])
    assert tips.loc["TST002", "severity"] == "tip"


def test_passed_contains_applicable_checks_with_no_finding(regression_report):
    """Checks that ran without reporting anything show up as passed."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize()
    assert "TST001" in set(result.frame(severity="passed")["code"])
    assert "TST001" not in set(result.frame(severity="issue")["code"])
    assert "TST001" not in set(result.frame(severity="tip")["code"])


def test_passed_excludes_ignored(regression_report):
    """Ignored codes are not listed as passed."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize(ignore=["TST001"])
    assert "TST001" not in set(result.frame(severity="passed")["code"])
    assert "TST001" not in set(result.frame(severity="issue")["code"])


def test_frame_severity_filter(regression_report):
    """`frame(severity=...)` returns only rows of the requested bucket."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize()

    issues_frame = result.frame(severity="issue")
    assert set(issues_frame["code"]) >= {"TST001"}
    assert all(issues_frame["severity"] == "issue")

    tips_frame = result.frame(severity="tip")
    assert set(tips_frame["code"]) >= {"TST002"}
    assert all(tips_frame["severity"] == "tip")

    passed_codes = set(result.frame(severity="passed")["code"])
    assert "TST001" not in passed_codes
    assert "TST002" not in passed_codes

    assert set(result.frame()["code"]) >= {"TST001", "TST002"}


def test_header_reports_all_counts(regression_report):
    """The header string reports issue, tip, passed and ignored counts."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize(ignore=["SKD001"])
    assert "issue(s)" in result._header
    assert "tip(s)" in result._header
    assert "passed" in result._header
    assert "1 ignored" in result._header


def test_html_has_three_tabs(regression_report):
    """The HTML repr contains one label per bucket with its count."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    html = regression_report.checks.summarize()._repr_html_()
    assert "Issues (" in html
    assert "Tips (" in html
    assert "Passed (" in html
