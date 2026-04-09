from pathlib import Path
from urllib.parse import urlparse

import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skore import Check, EstimatorReport, configuration, evaluate
from skore._sklearn._diagnostic import DiagnosticDisplay, get_issue_documentation_url
from skore._sklearn._diagnostic.utils import DiagnosticNotApplicable


def mock_issue(report):
    return (
        {
            "SKD001": {
                "title": "Mock title",
                "docs_url": "skd001-overfitting",
                "explanation": "Mock overfitting detected.",
            }
        },
        {"SKD001"},
    )


def test_diagnose_detects_overfitting(regression_data):
    """Check that the overfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
    result = report.diagnose()
    assert "SKD001" in result.issues
    assert "2/2 default predictive metrics" in result.issues["SKD001"]["explanation"]


def test_diagnose_detects_underfitting(regression_data):
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    result = report.diagnose()
    assert "SKD002" in result.issues
    assert "2/2 comparable metrics" in result.issues["SKD002"]["explanation"]


def test_diagnose_ignore(monkeypatch, regression_data):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(EstimatorReport, "_get_issues", mock_issue)
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
                check.run(report)


def test_diagnose_no_issues(monkeypatch, regression_data):
    """Check that no issues are detected when checks pass."""
    X, y = regression_data
    monkeypatch.setattr(
        EstimatorReport,
        "_get_issues",
        lambda report: ({}, {"SKD001", "SKD002"}),
    )
    report = evaluate(LinearRegression(), X, y)
    result = report.diagnose()
    assert result.issues == {}


def test_diagnose_result_repr(monkeypatch, regression_data):
    """Check that the diagnostic result has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_issues", mock_issue)
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
    monkeypatch.setattr(EstimatorReport, "_get_issues", mock_issue)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    assert "SKD001" in report.diagnose().issues
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in report.diagnose().issues


def test_diagnose_documentation_url_points_to_existing_rst():
    """Check that the URL in get_issue_documentation_url maps to a real RST file."""
    url = urlparse(get_issue_documentation_url(docs_anchor="placeholder"))
    # url.path is e.g. "/dev/user_guide/automatic_diagnostic.html"
    # strip version prefix and convert .html -> .rst
    rst_rel_path = "/".join(url.path.split("/")[2:]).replace(".html", ".rst")
    rst_path = Path(__file__).parents[5] / "sphinx" / rst_rel_path
    assert rst_path.is_file()


def test_diagnose_reuses_cached_results(monkeypatch, regression_data):
    """Check that check results are cached and reused."""
    calls = 0
    original_run = Check.run

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "run", counting_run)
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

    check = Check(
        lambda report: "Something was found.",
        "CUSTOM001",
        "Custom issue",
        "estimator",
        "custom001",
    )
    report.add_checks([check])
    result = report.diagnose()
    assert "CUSTOM001" in result.issues
    assert result.issues["CUSTOM001"]["title"] == "Custom issue"
    assert result.issues["CUSTOM001"]["docs_url"] == "custom001"
    assert result.issues["CUSTOM001"]["explanation"] == "Something was found."


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_data):
    """Check that add_checks does not re-run already cached built-in checks."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    calls = 0
    original_run = Check.run

    def counting_run(self, rpt):
        nonlocal calls
        calls += 1
        return original_run(self, rpt)

    monkeypatch.setattr(Check, "run", counting_run)

    report.diagnose()
    calls_after_first = calls

    def my_check(report):
        return "Details."

    custom = Check(my_check, "CUSTOM002", "Another issue", "estimator")
    report.add_checks([custom])
    report.diagnose()

    # Only the new custom check should have run, not the built-in ones again
    assert calls == calls_after_first + 1


def test_add_checks_docs_url_full(regression_data):
    """Check that a full https docs_url is preserved as-is in frame()."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def check_with_url(report):
        return "Has a URL."

    check = Check(
        check_with_url, "URL001", "URL test", "estimator", "https://example.com/my-doc"
    )
    report.add_checks([check])
    result = report.diagnose()
    frame = result.frame()
    row = frame.query("code == 'URL001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_data):
    """Check that missing docs_url results in None in the frame."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def check_no_url(report):
        return "No docs_url provided."

    check = Check(check_no_url, "NOURL001", "No URL", "estimator")
    report.add_checks([check])
    result = report.diagnose()
    frame = result.frame()
    row = frame[frame["code"] == "NOURL001"]
    assert row["documentation_url"].iloc[0] is None


def test_check_invalid_report_type():
    """Check that Check raises ValueError for unsupported report_type."""

    def my_check(report):
        return None

    with pytest.raises(ValueError, match="report_type should be one of"):
        Check(my_check, "X", "Title", "invalid")
