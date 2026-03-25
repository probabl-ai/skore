from unittest.mock import patch

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from skore import EstimatorReport, configuration, evaluate
from skore._sklearn._diagnostics.base import DiagnosticResult, DiagnosticResults


def test_diagnose_detects_overfitting():
    """Check the overfitting diagnostics are detected."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=2,
        n_redundant=0,
        class_sep=0.4,
        flip_y=0.25,
        random_state=0,
    )
    report = evaluate(DecisionTreeClassifier(random_state=0), X, y, splitter=0.5)
    messages = report.diagnose()
    assert any("[SKD001]" in message for message in messages)


def test_diagnose_detects_underfitting():
    """Check the underfitting diagnostics are detected."""
    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    report = EstimatorReport(
        DummyClassifier(strategy="uniform", random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    messages = report.diagnose()
    assert any("[SKD002]" in message for message in messages)


def test_diagnose_ignore(monkeypatch, regression_train_test_split):
    """Check the diagnostics are ignored when ignore is passed."""
    diagnostic = DiagnosticResult(
        code="SKD001",
        title="Mock overfitting",
        docs_anchor="skd001-overfitting",
        explanation="Mock overfitting detected.",
    )
    monkeypatch.setattr(
        EstimatorReport,
        "_compute_diagnostics",
        lambda self: ([diagnostic], {"SKD001", "SKD002"}),
    )
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    messages = report.diagnose(ignore=["SKD001"])
    assert all("[SKD001]" not in message for message in messages)
    assert messages.diagnostics == []


def test_diagnose_empty_when_train_data_missing(regression_data):
    """Check the diagnostics are empty when the train data is missing."""
    X, y = regression_data
    estimator = LogisticRegression(max_iter=1_000).fit(X, y > y.mean())
    report = EstimatorReport(estimator, X_test=X, y_test=y > y.mean())
    messages = report.diagnose()
    assert messages == ["No issues were detected in your report!"]
    assert messages.diagnostics == []


def test_diagnose_no_issues(monkeypatch, regression_train_test_split):
    """Check the diagnostics are empty when no issues are detected."""
    monkeypatch.setattr(
        EstimatorReport,
        "_compute_diagnostics",
        lambda self: ([], {"SKD001", "SKD002"}),
    )
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    messages = report.diagnose()
    assert messages == ["No issues were detected in your report!"]
    assert messages.diagnostics == []


def test_diagnose_called_on_init(monkeypatch, regression_train_test_split):
    """Check the diagnostics are called on init."""
    calls = []

    def _compute_diagnostics(self):
        calls.append(True)
        return [], set()

    monkeypatch.setattr(EstimatorReport, "_compute_diagnostics", _compute_diagnostics)
    X_train, X_test, y_train, y_test = regression_train_test_split
    EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        diagnose=True,
    )
    assert calls == [True]


def test_diagnose_result_has_repr(monkeypatch, regression_train_test_split):
    """Check the diagnostics result has a repr."""
    diagnostic = DiagnosticResult(
        code="SKD999",
        title="Mock issue",
        docs_anchor="skd001-overfitting",
        explanation="Mock issue for repr rendering.",
    )
    monkeypatch.setattr(
        EstimatorReport,
        "_compute_diagnostics",
        lambda self: ([diagnostic], {"SKD999"}),
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
    assert isinstance(results, DiagnosticResults)
    assert "Diagnostics:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/diagnostics.html#" in bundle["text/html"]


def test_diagnose_uses_global_ignore(monkeypatch, regression_data):
    """Check the diagnostics are ignored when global ignore is set."""
    diagnostic = DiagnosticResult(
        code="SKD001",
        title="Mock overfitting",
        docs_anchor="skd001-overfitting",
        explanation="Mock overfitting detected.",
    )
    monkeypatch.setattr(
        EstimatorReport,
        "_compute_diagnostics",
        lambda self: ([diagnostic], {"SKD001", "SKD002"}),
    )
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2)
    assert any(d.code == "SKD001" for d in report.diagnose().diagnostics)
    with configuration(ignore_diagnostics=["SKD001"]):
        assert all(d.code != "SKD001" for d in report.diagnose().diagnostics)


def test_diagnose_follows_global_config_default(regression_data):
    """Check the diagnostics are displayed on init in console by default."""
    X, y = regression_data
    with patch.object(EstimatorReport, "_display_diagnose_results") as display_mock:
        evaluate(LinearRegression(), X, y, splitter=0.2)
    display_mock.assert_not_called()
    with (
        patch.object(EstimatorReport, "_display_diagnose_results") as display_mock,
        configuration(diagnose=True),
    ):
        evaluate(LinearRegression(), X, y, splitter=0.2)
    display_mock.assert_called_once()


def test_diagnose_reuses_cached_diagnostics(monkeypatch, regression_data):
    """Check the diagnostics are reused across splits."""
    calls = 0
    original = EstimatorReport._compute_diagnostics

    def _compute_diagnostics(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(EstimatorReport, "_compute_diagnostics", _compute_diagnostics)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.2, diagnose=True)
    calls_after_init = calls
    report.diagnose()
    assert calls == calls_after_init
