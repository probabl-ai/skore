from pathlib import Path
from urllib.parse import urlparse

import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from skore import EstimatorReport, configuration, evaluate
from skore._sklearn._diagnostic import DiagnosticDisplay, get_issue_documentation_url
from skore._sklearn._diagnostic.utils import DiagnosticNotApplicable


def test_diagnose_detects_overfitting():
    """Check that the overfitting issue is detected."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=2,
        n_redundant=0,
        class_sep=0.4,
        flip_y=0.25,
        random_state=0,
    )
    report = evaluate(
        DecisionTreeClassifier(random_state=0), X, y, splitter=0.5, pos_label=1
    )
    result = report.diagnose()
    assert "SKD001" in result.issues
    assert "6/6 default predictive metrics" in result.issues["SKD001"]["explanation"]


def test_diagnose_detects_underfitting():
    """Check that the underfitting issue is detected."""
    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    report = EstimatorReport(
        DummyClassifier(strategy="prior"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label=1,
    )
    result = report.diagnose()
    assert "SKD002" in result.issues
    assert "6/6 comparable metrics" in result.issues["SKD002"]["explanation"]


def test_diagnose_ignore(monkeypatch, regression_train_test_split):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(
        EstimatorReport,
        "_run_checks",
        lambda self: (
            {
                "SKD001": {
                    "title": "Mock overfitting",
                    "docs_url": "skd001-overfitting",
                    "explanation": "Mock overfitting detected.",
                }
            },
            {"SKD001", "SKD002"},
        ),
    )
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    result = report.diagnose(ignore=["SKD001"])
    assert "SKD001" not in result.issues
    assert result.issues == {}


def test_diagnose_empty_when_train_data_missing(regression_data):
    """Check that no issues are detected when the train data is missing."""
    X, y = regression_data
    estimator = LogisticRegression(max_iter=1_000).fit(X, y > y.mean())
    report = EstimatorReport(estimator, X_test=X, y_test=y > y.mean())
    result = report.diagnose()
    assert result.issues == {}


def test_diagnose_no_issues(monkeypatch, regression_train_test_split):
    """Check that no issues are detected when checks pass."""
    monkeypatch.setattr(
        EstimatorReport,
        "_run_checks",
        lambda self: ({}, {"SKD001", "SKD002"}),
    )
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    result = report.diagnose()
    assert result.issues == {}


def test_diagnose_result_has_repr(monkeypatch, regression_train_test_split):
    """Check that the diagnostic result has a repr."""
    monkeypatch.setattr(
        EstimatorReport,
        "_run_checks",
        lambda self: (
            {
                "SKD999": {
                    "title": "Mock issue",
                    "docs_url": "skd001-overfitting",
                    "explanation": "Mock issue for repr rendering.",
                }
            },
            {"SKD999"},
        ),
    )
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    assert "Diagnostic:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/automatic_diagnostic.html#" in bundle["text/html"]


def test_diagnose_uses_global_ignore(monkeypatch, regression_data):
    """Check that checks are ignored when global ignore is set."""
    monkeypatch.setattr(
        EstimatorReport,
        "_run_checks",
        lambda self: (
            {
                "SKD001": {
                    "title": "Mock overfitting",
                    "docs_url": "skd001-overfitting",
                    "explanation": "Mock overfitting detected.",
                }
            },
            {"SKD001", "SKD002"},
        ),
    )
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
    original = EstimatorReport._run_checks

    def _run_checks_wrapper(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(EstimatorReport, "_run_checks", _run_checks_wrapper)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    report.diagnose()
    calls_after_first = calls
    report.diagnose()
    assert calls == calls_after_first


# ---- Custom check tests ----


def _make_report(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    return EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def test_add_checks_runs_custom_check(regression_train_test_split):
    """Check that add_checks runs the custom check and includes its issue."""
    report = _make_report(regression_train_test_split)

    def my_check(report):
        return {
            "CUSTOM001": {
                "title": "Custom issue",
                "explanation": "Something was found.",
            }
        }

    result = report.add_checks([("CUSTOM001", my_check)])
    assert "CUSTOM001" in result.issues
    assert result.issues["CUSTOM001"]["title"] == "Custom issue"


def test_add_checks_reuses_builtin_cache(regression_train_test_split):
    """Check that add_checks does not re-run already cached built-in checks."""
    report = _make_report(regression_train_test_split)

    report.diagnose()
    cached_before = dict(report._check_results_cache)

    def my_check(report):
        return {
            "CUSTOM002": {
                "title": "Another issue",
                "explanation": "Details.",
            }
        }

    report.add_checks([("CUSTOM002", my_check)])

    for code in ("SKD001", "SKD002"):
        if code in cached_before:
            assert report._check_results_cache[code] is cached_before[code]


def test_add_checks_validates_entry(regression_train_test_split):
    """Check that add_checks raises TypeError for malformed entries."""
    report = _make_report(regression_train_test_split)

    with pytest.raises(TypeError, match="\\(code, callable\\) tuple"):
        report.add_checks(["not_a_tuple"])

    with pytest.raises(TypeError, match="\\(code, callable\\) tuple"):
        report.add_checks([(123, lambda r: {})])

    with pytest.raises(TypeError, match="\\(code, callable\\) tuple"):
        report.add_checks([("CODE", "not_callable")])


def test_add_checks_validates_return_value(regression_train_test_split):
    """Check that add_checks raises TypeError for bad return value."""
    report = _make_report(regression_train_test_split)

    def bad_check(report):
        return "not a dict"

    with pytest.raises(TypeError, match="dict"):
        report.add_checks([("BAD", bad_check)])


def test_add_checks_not_applicable(regression_train_test_split):
    """Check that DiagnosticNotApplicable is silently skipped."""
    report = _make_report(regression_train_test_split)

    def inapplicable_check(report):
        raise DiagnosticNotApplicable()

    result = report.add_checks([("NA001", inapplicable_check)])
    assert isinstance(result, DiagnosticDisplay)


def test_add_checks_docs_url_full(regression_train_test_split):
    """Check that a full https docs_url is preserved as-is in frame()."""
    report = _make_report(regression_train_test_split)

    def check_with_url(report):
        return {
            "URL001": {
                "title": "URL test",
                "explanation": "Has a URL.",
                "docs_url": "https://example.com/my-doc",
            }
        }

    result = report.add_checks([("URL001", check_with_url)])
    frame = result.frame()
    row = frame[frame["code"] == "URL001"]
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_train_test_split):
    """Check that missing docs_url results in None in the frame."""
    report = _make_report(regression_train_test_split)

    def check_no_url(report):
        return {
            "NOURL001": {
                "title": "No URL",
                "explanation": "No docs_url provided.",
            }
        }

    result = report.add_checks([("NOURL001", check_no_url)])
    frame = result.frame()
    row = frame[frame["code"] == "NOURL001"]
    assert row["documentation_url"].iloc[0] is None


def test_add_checks_invalid_level(regression_train_test_split):
    """Check that add_checks raises ValueError for unsupported level."""
    report = _make_report(regression_train_test_split)

    def my_check(report):
        return {}

    with pytest.raises(ValueError, match="level="):
        report.add_checks([("X", my_check)], level="cross-validation")
