from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skore import CrossValidationReport, EstimatorReport, configuration


def test_diagnose_aggregates_overfitting_across_splits():
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=2,
        n_redundant=0,
        class_sep=0.4,
        flip_y=0.25,
        random_state=0,
    )
    report = CrossValidationReport(
        DecisionTreeClassifier(random_state=0), X, y, splitter=5
    )
    messages = report.diagnose()
    assert any(
        "[SKD001]" in message and "issue detected" in message for message in messages
    )
    assert any("evaluated splits" in message for message in messages)


def test_diagnose_aggregates_underfitting_across_splits():
    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    report = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0),
        X,
        y,
        splitter=5,
    )
    results = report.diagnose()
    underfitting = next(
        diagnostic for diagnostic in results.diagnostics if diagnostic.code == "SKD002"
    )
    assert underfitting.evaluated
    assert "evaluated split" in underfitting.explanation
    if underfitting.is_issue:
        assert any("[SKD002]" in message for message in results)
    else:
        assert results == ["No issues were detected in your report!"]


def test_diagnose_ignore(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    messages = report.diagnose(ignore=["SKD001"])
    assert all("[SKD001]" not in message for message in messages)


def test_diagnose_called_on_init(monkeypatch, binary_classification_data):
    calls = []

    def _collect_diagnostics(self):
        calls.append(True)
        return []

    monkeypatch.setattr(
        CrossValidationReport, "_collect_diagnostics", _collect_diagnostics
    )
    X, y = binary_classification_data
    CrossValidationReport(LogisticRegression(), X, y, splitter=2, diagnose=True)
    assert calls == [True]


def test_diagnose_result_has_repr(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    results = report.diagnose()
    assert isinstance(results, list)
    assert "Diagnostics:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle


def test_diagnose_reuses_split_cached_diagnostics(
    monkeypatch, binary_classification_data
):
    calls = 0
    original = EstimatorReport._collect_diagnostics

    def _collect_diagnostics(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(EstimatorReport, "_collect_diagnostics", _collect_diagnostics)
    X, y = binary_classification_data
    report = CrossValidationReport(
        LogisticRegression(), X, y, splitter=3, diagnose=True
    )
    calls_after_init = calls
    report.diagnose()
    assert calls == calls_after_init


def test_diagnose_uses_global_ignore(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    assert any(
        diagnostic.code == "SKD001" for diagnostic in report.diagnose().diagnostics
    )
    with configuration(ignore_diagnostics=["SKD001"]):
        assert all(
            diagnostic.code != "SKD001" for diagnostic in report.diagnose().diagnostics
        )


def test_diagnose_follows_global_config_default(binary_classification_data):
    X, y = binary_classification_data
    with configuration(diagnose=True):
        report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    assert len(report._diagnostics_cache) > 0
