from pathlib import Path
from urllib.parse import urlparse

import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from skore import Check, EstimatorReport, configuration, evaluate
from skore._sklearn._diagnostic import DiagnosticDisplay
from skore._sklearn._diagnostic.base import _get_issue_documentation_url
from skore._sklearn._diagnostic.utils import DiagnosticNotApplicable


def mock_issue(report):
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


def test_diagnose_detects_overfitting(regression_data):
    """Check that the overfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
    result = report.diagnose()
    assert "SKD001" in result.issues
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2

    assert (
        f"for {n_metrics}/{n_metrics} default predictive metrics"
        in result.issues["SKD001"]["explanation"]
    )


def test_diagnose_detects_underfitting(regression_data):
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    result = report.diagnose()
    assert "SKD002" in result.issues
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in result.issues["SKD002"]["explanation"]
    )


@pytest.mark.parametrize(
    "weights, code", [([0.9, 0.1], "SKD004"), ([0.9, 0.05, 0.05], "SKD005")]
)
def test_diagnose_detects_high_class_imbalance(weights, code):
    """Check that the high class imbalance issue is detected."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    result = report.diagnose()
    assert code not in result.issues

    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        weights=weights,
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    result = report.diagnose()
    assert code in result.issues
    assert "Accuracy should not be used alone" in result.issues[code]["explanation"]


def test_diagnose_detects_unscaled_coefficients(regression_data):
    """Check that the unscaled coefficients issue is detected."""
    X, y = regression_data
    result = evaluate(LinearRegression(), X, y).diagnose()
    assert "SKD006" in result.tips
    assert "coefficients" in result.tips["SKD006"]["explanation"]


def test_diagnose_ignore(monkeypatch, regression_data):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(EstimatorReport, "_get_findings", mock_issue)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    result = report.diagnose(ignore=["SKD001"])
    assert "SKD001" not in result.issues
    assert result.issues == {}


def test_exception_when_train_data_missing(regression_train_test_split):
    """Check that an exception is raised when the train data is missing."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    for check in report._checks_registry:
        if check.code in ["SKD001", "SKD002"]:
            with pytest.raises(DiagnosticNotApplicable):
                check.check_function(report)


def test_diagnose_no_issues(monkeypatch, regression_data):
    """Check that no issues are detected when checks pass."""
    X, y = regression_data
    monkeypatch.setattr(
        EstimatorReport,
        "_get_findings",
        lambda report: ({}, {"SKD001", "SKD002"}),
    )
    report = evaluate(LinearRegression(), X, y)
    result = report.diagnose()
    assert result.issues == {}


def test_diagnose_result_repr(monkeypatch, regression_data):
    """Check that the diagnostic result has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_findings", mock_issue)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    elements = [
        "Diagnostic:",
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
    assert "user_guide/automatic_diagnostic.html#" in bundle["text/html"]


def test_diagnose_uses_global_ignore(monkeypatch, regression_data):
    """Check that checks are ignored when global ignore is set."""
    monkeypatch.setattr(EstimatorReport, "_get_findings", mock_issue)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    assert "SKD001" in report.diagnose().issues
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in report.diagnose().issues


def test_diagnose_documentation_url_points_to_existing_rst():
    """Check that the URL in _get_issue_documentation_url maps to a real RST file."""
    url = urlparse(_get_issue_documentation_url(mock_issue(None)[0]["SKD001"]))
    # url.path is e.g. "/dev/user_guide/automatic_diagnostic.html"
    # strip version prefix and convert .html -> .rst
    rst_rel_path = "/".join(url.path.split("/")[2:]).replace(".html", ".rst")
    rst_path = Path(__file__).parents[5] / "sphinx" / rst_rel_path
    assert rst_path.is_file()


def test_diagnose_reuses_cached_results(monkeypatch, regression_data):
    """Check that check results are cached and reused."""
    calls = 0
    original_run = Check.check_function

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "check_function", counting_run)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    report.diagnose()
    calls_after_first = calls
    report.diagnose()
    assert calls == calls_after_first


def test_add_checks_runs_custom_check(regression_data):
    """Check that add_checks runs the custom check and includes its issue."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    report.add_checks([MockCheck(has_issue=True)])
    result = report.diagnose()
    assert "TST001" in result.issues
    assert result.issues["TST001"]["title"] == "Test issue"
    assert result.issues["TST001"]["docs_url"] == "tst001"
    assert result.issues["TST001"]["explanation"] == "Something was found."


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_data):
    """Check that add_checks does not re-run already cached built-in checks."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.diagnose()

    for check in report._checks_registry:
        monkeypatch.setattr(
            check, "check_function", lambda report: pytest.fail("re-ran cached check")
        )

    report.add_checks([MockCheck(has_issue=True)])
    report.diagnose()


def test_add_checks_docs_url_full(regression_data):
    """Check that a full https docs_url is preserved as-is in frame()."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    check = MockCheck(has_issue=True, docs_url="https://example.com/my-doc")
    report.add_checks([check])
    result = report.diagnose()
    frame = result.frame()
    row = frame.query("code == 'TST001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_data):
    """Check that missing docs_url results in None in the frame."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    check = MockCheck(has_issue=True, docs_url=None)
    report.add_checks([check])
    result = report.diagnose()
    frame = result.frame()
    row = frame[frame["code"] == "TST001"]
    assert row["documentation_url"].isna().all()


def test_check_invalid_report_type(regression_data):
    """Check that Check raises ValueError for unsupported report_type."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    check = MockCheck(has_issue=False, report_type="invalid")
    with pytest.raises(ValueError, match="report_type should be one of"):
        report.add_checks([check])


def test_check_invalid_protocol(regression_data):
    """Check that Check raises ValueError for unsupported protocol."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    class InvalidCheck:
        code = "INVALID001"
        title = "Invalid issue"
        report_type = "estimator"
        docs_url = "invalid001"

    with pytest.raises(ValueError, match="does not implement the Check protocol."):
        report.add_checks([InvalidCheck()])


def test_diagnose_custom_metric(binary_classification_data):
    """Check that diagnose works with custom metrics in the report."""
    X, y = binary_classification_data
    report = evaluate(DummyClassifier(), X, y, pos_label=1)
    report.metrics.add("f1")
    result = report.diagnose()
    assert "SKD002" in result.issues
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in result.issues["SKD002"]["explanation"]
    )


class TipCheck(Check):
    code = "TST002"
    title = "Tip check"
    report_type = "estimator"
    docs_url = "tst_tip"
    severity = "tip"

    def check_function(self, report):
        return "Be careful about this."


def test_diagnose_tip_goes_to_tips_not_issues(regression_data):
    """A check with severity='tip' is routed to `.tips`, not `.issues`."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([TipCheck()])
    result = report.diagnose()
    assert "TST002" in result.tips
    assert "TST002" not in result.issues
    assert result.tips["TST002"]["severity"] == "tip"


def test_diagnose_passed_contains_applicable_checks_with_no_finding(regression_data):
    """Checks that ran without reporting anything show up in `.passed`."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([MockCheck(has_issue=False)])
    result = report.diagnose()
    assert "TST001" in result.passed
    assert "TST001" not in result.issues
    assert "TST001" not in result.tips


def test_diagnose_passed_excludes_ignored(regression_data):
    """Ignored codes are not listed as passed."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([MockCheck(has_issue=False)])
    result = report.diagnose(ignore=["TST001"])
    assert "TST001" not in result.passed
    assert "TST001" not in result.issues


def test_diagnose_frame_severity_filter(regression_data):
    """`frame(severity=...)` returns only rows of the requested bucket."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([MockCheck(has_issue=True), TipCheck()])
    result = report.diagnose()

    issues_frame = result.frame(severity="issue")
    assert set(issues_frame["code"]) >= {"TST001"}
    assert all(issues_frame["severity"] == "issue")

    tips_frame = result.frame(severity="tip")
    assert set(tips_frame["code"]) >= {"TST002"}
    assert all(tips_frame["severity"] == "tip")

    passed_frame = result.frame(severity="passed")
    assert "TST001" not in set(passed_frame["code"])
    assert "TST002" not in set(passed_frame["code"])
    assert passed_frame["explanation"].isna().all()

    all_frame = result.frame()
    assert set(all_frame["code"]) >= {"TST001", "TST002"}


def test_diagnose_header_reports_all_counts(regression_data):
    """The header string reports issue, tip, passed and ignored counts."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([MockCheck(has_issue=True), TipCheck()])
    result = report.diagnose(ignore=["SKD001"])
    assert "issue(s)" in result.header
    assert "tip(s)" in result.header
    assert "passed" in result.header
    assert "1 ignored" in result.header


def test_diagnose_html_has_three_tabs(regression_data):
    """The HTML repr contains one label per bucket with its count."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    report.add_checks([MockCheck(has_issue=True), TipCheck()])
    html = report.diagnose()._repr_html_()
    assert "Issues (" in html
    assert "Tips (" in html
    assert "Passed (" in html
