from pathlib import Path
from urllib.parse import urlparse

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skore import EstimatorReport, configuration, evaluate
from skore._sklearn._diagnostic import DiagnosticDisplay, get_issue_documentation_url


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


def test_diagnose_detects_overfitting_regression(regression_data):
    """Check that the overfitting issue is detected for regression with 4 metrics."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
    result = report.diagnose()
    assert "SKD001" in result.issues
    assert "4/4 default predictive metrics" in result.issues["SKD001"]["explanation"]


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


def test_diagnose_detects_underfitting_regression(regression_data):
    """Check that the underfitting issue is detected for regression with 4 metrics."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    result = report.diagnose()
    assert "SKD002" in result.issues
    assert "4/4 comparable metrics" in result.issues["SKD002"]["explanation"]


def test_diagnose_ignore(monkeypatch, regression_train_test_split):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(
        EstimatorReport,
        "_run_checks",
        lambda self: (
            {
                "SKD001": {
                    "title": "Mock overfitting",
                    "docs_anchor": "skd001-overfitting",
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
                    "docs_anchor": "skd001-overfitting",
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
                    "docs_anchor": "skd001-overfitting",
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
