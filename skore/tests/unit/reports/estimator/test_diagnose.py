from unittest.mock import Mock

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from skore import EstimatorReport, configuration
from skore._sklearn._diagnostics.base import DiagnosticResult


def test_diagnose_detects_overfitting():
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=2,
        n_redundant=0,
        class_sep=0.4,
        flip_y=0.25,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    report = EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    messages = report.diagnose()
    assert any(
        "[SKD001]" in message and "issue detected" in message for message in messages
    )


def test_diagnose_detects_underfitting():
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
    assert any(
        "[SKD002]" in message and "issue detected" in message for message in messages
    )


def test_diagnose_ignore(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    messages = report.diagnose(ignore=["SKD001"])
    assert all("[SKD001]" not in message for message in messages)


def test_diagnose_not_evaluated_when_train_data_missing(regression_data):
    X, y = regression_data
    estimator = LogisticRegression(max_iter=1_000).fit(X, y > y.mean())
    report = EstimatorReport(estimator, X_test=X, y_test=y > y.mean())
    messages = report.diagnose()
    assert messages == ["No issues were detected in your report!"]
    assert any(
        diagnostic.code == "SKD001" and not diagnostic.evaluated
        for diagnostic in messages.diagnostics
    )
    assert any(
        diagnostic.code == "SKD002" and not diagnostic.evaluated
        for diagnostic in messages.diagnostics
    )


def test_diagnose_hides_non_issue_diagnostics(monkeypatch, regression_train_test_split):
    diagnostic = DiagnosticResult(
        code="SKD999",
        title="Mock diagnostic",
        kind="info",
        docs_anchor="skd001-overfitting",
        explanation="No issue detected for this mock diagnostic.",
        is_issue=False,
        evaluated=True,
    )
    monkeypatch.setattr(
        EstimatorReport,
        "_collect_diagnostics",
        lambda self: [diagnostic],
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
    assert messages.diagnostics == (diagnostic,)


def test_diagnose_called_on_init(monkeypatch, regression_train_test_split):
    calls = []

    def _collect_diagnostics(self):
        calls.append(True)
        return []

    monkeypatch.setattr(EstimatorReport, "_collect_diagnostics", _collect_diagnostics)
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
    diagnostic = DiagnosticResult(
        code="SKD999",
        title="Mock issue",
        kind="info",
        docs_anchor="skd001-overfitting",
        explanation="Mock issue for repr rendering.",
        is_issue=True,
        evaluated=True,
    )
    monkeypatch.setattr(
        EstimatorReport,
        "_collect_diagnostics",
        lambda self: [diagnostic],
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
    assert isinstance(results, list)
    assert "Diagnostics:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/diagnostics.html#" in bundle["text/html"]


def test_diagnose_displayed_on_init_notebook(monkeypatch, regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    mock_sphinx = Mock(return_value=False)
    mock_notebook = Mock(return_value=True)
    mock_display = Mock()
    monkeypatch.setattr(
        "skore._sklearn._base.is_environment_sphinx_build",
        mock_sphinx,
    )
    monkeypatch.setattr(
        "skore._sklearn._base.is_environment_notebook_like",
        mock_notebook,
    )
    monkeypatch.setattr("IPython.display.display", mock_display)
    EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        diagnose=True,
    )
    mock_display.assert_called_once()


def test_diagnose_uses_global_ignore(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert any(
        diagnostic.code == "SKD001" for diagnostic in report.diagnose().diagnostics
    )
    with configuration(ignore_diagnostics=["SKD001"]):
        assert all(
            diagnostic.code != "SKD001" for diagnostic in report.diagnose().diagnostics
        )


def test_diagnose_follows_global_config_default(
    logistic_binary_classification_with_train_test,
):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    with configuration(diagnose=True):
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    assert hasattr(report, "_latest_diagnostics_")
    assert len(report._latest_diagnostics_) > 0
