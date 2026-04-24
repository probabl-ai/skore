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
from skore._sklearn._diagnostic.utils import CheckNotApplicable


@pytest.fixture
def regression_report(regression_data):
    X, y = regression_data
    return evaluate(LinearRegression(), X, y)


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
    issues = report.diagnose().frame(severity="issue").set_index("code")
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
    issues = report.diagnose().frame(severity="issue").set_index("code")
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
    result = report.diagnose()
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
    issues = report.diagnose().frame(severity="issue").set_index("code")
    assert code in issues.index
    assert "Accuracy should not be used alone" in issues.loc[code, "explanation"]


def test_skd006_detects_coefficient_interpretation(regression_data):
    """Check that the coefficient interpretation tip is emitted."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    tips = report.diagnose().frame(severity="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y)
    tips = report.diagnose().frame(severity="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


def test_ignore_checks(monkeypatch, regression_report):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert regression_report.diagnose(ignore=["SKD001"]).frame(severity="issue").empty


def test_exception_when_train_data_missing(regression_train_test_split):
    """Check that an exception is raised when the train data is missing."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    for check in report._checks_registry:
        if check.code in ["SKD001", "SKD002"]:
            with pytest.raises(CheckNotApplicable):
                check.check_function(report)


def test_diagnose_no_issues(monkeypatch, regression_report):
    """Check that no issues are detected when checks pass."""
    monkeypatch.setattr(
        EstimatorReport,
        "_get_results",
        lambda report, ignored_codes: ({}, {"SKD001", "SKD002"}),
    )
    assert regression_report.diagnose().frame(severity="issue").empty


def test_diagnostic_result_repr(monkeypatch, regression_report):
    """Check that the diagnostic result has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    results = regression_report.diagnose()
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


def test_global_ignore(monkeypatch, regression_report):
    """Check that checks are ignored when global ignore is set."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert "SKD001" in set(regression_report.diagnose().frame(severity="issue")["code"])
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in set(
            regression_report.diagnose().frame(severity="issue")["code"]
        )


def test_documentation_url_points_to_existing_rst():
    """Check that the URL in _get_issue_documentation_url maps to a real RST file."""
    url = urlparse(_get_issue_documentation_url(mock_issue(None, set())[0]["SKD001"]))
    # url.path is e.g. "/dev/user_guide/automatic_diagnostic.html"
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
    regression_report.diagnose()
    calls_after_first = calls
    regression_report.diagnose()
    assert calls == calls_after_first


def test_add_checks_runs_custom_check(regression_report):
    """Check that add_checks runs the custom check and includes its issue."""
    regression_report.add_checks([MockCheck(has_issue=True)])
    issues = regression_report.diagnose().frame(severity="issue").set_index("code")
    assert "TST001" in issues.index
    assert issues.loc["TST001", "title"] == "Test issue"
    assert issues.loc["TST001", "documentation_url"].endswith("#tst001")
    assert issues.loc["TST001", "explanation"] == "Something was found."


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_report):
    """Check that add_checks does not re-run already cached built-in checks."""
    regression_report.diagnose()

    for check in regression_report._checks_registry:
        monkeypatch.setattr(
            check, "check_function", lambda report: pytest.fail("re-ran cached check")
        )

    regression_report.add_checks([MockCheck(has_issue=True)])
    regression_report.diagnose()


def test_add_checks_docs_url_full(regression_report):
    """Check that a full https docs_url is preserved as-is in frame()."""
    check = MockCheck(has_issue=True, docs_url="https://example.com/my-doc")
    regression_report.add_checks([check])
    result = regression_report.diagnose()
    frame = result.frame()
    row = frame.query("code == 'TST001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_report):
    """Check that missing docs_url results in None in the frame."""
    check = MockCheck(has_issue=True, docs_url=None)
    regression_report.add_checks([check])
    result = regression_report.diagnose()
    frame = result.frame()
    row = frame[frame["code"] == "TST001"]
    assert row["documentation_url"].isna().all()


def test_check_invalid_report_type(regression_report):
    """Check that Check raises ValueError for unsupported report_type."""
    check = MockCheck(has_issue=False, report_type="invalid")
    with pytest.raises(ValueError, match="report_type should be one of"):
        regression_report.add_checks([check])


def test_check_invalid_protocol(regression_report):
    """Check that Check raises ValueError for unsupported protocol."""

    class InvalidCheck:
        code = "INVALID001"
        title = "Invalid issue"
        report_type = "estimator"
        docs_url = "invalid001"

    with pytest.raises(ValueError, match="does not implement the Check protocol."):
        regression_report.add_checks([InvalidCheck()])


def test_diagnose_custom_metric(binary_classification_data):
    """Check that diagnose works with custom metrics in the report."""
    X, y = binary_classification_data
    report = evaluate(DummyClassifier(), X, y, pos_label=1)
    report.metrics.add("f1")
    issues = report.diagnose().frame(severity="issue").set_index("code")
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
    regression_report.add_checks([TipCheck()])
    result = regression_report.diagnose()
    tips = result.frame(severity="tip").set_index("code")
    assert "TST002" in tips.index
    assert "TST002" not in set(result.frame(severity="issue")["code"])
    assert tips.loc["TST002", "severity"] == "tip"


def test_passed_contains_applicable_checks_with_no_finding(regression_report):
    """Checks that ran without reporting anything show up as passed."""
    regression_report.add_checks([MockCheck(has_issue=False)])
    result = regression_report.diagnose()
    assert "TST001" in set(result.frame(severity="passed")["code"])
    assert "TST001" not in set(result.frame(severity="issue")["code"])
    assert "TST001" not in set(result.frame(severity="tip")["code"])


def test_passed_excludes_ignored(regression_report):
    """Ignored codes are not listed as passed."""
    regression_report.add_checks([MockCheck(has_issue=False)])
    result = regression_report.diagnose(ignore=["TST001"])
    assert "TST001" not in set(result.frame(severity="passed")["code"])
    assert "TST001" not in set(result.frame(severity="issue")["code"])


def test_frame_severity_filter(regression_report):
    """`frame(severity=...)` returns only rows of the requested bucket."""
    regression_report.add_checks([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.diagnose()

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
    regression_report.add_checks([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.diagnose(ignore=["SKD001"])
    assert "issue(s)" in result.header
    assert "tip(s)" in result.header
    assert "passed" in result.header
    assert "1 ignored" in result.header


def test_html_has_three_tabs(regression_report):
    """The HTML repr contains one label per bucket with its count."""
    regression_report.add_checks([MockCheck(has_issue=True), TipCheck()])
    html = regression_report.diagnose()._repr_html_()
    assert "Issues (" in html
    assert "Tips (" in html
    assert "Passed (" in html
