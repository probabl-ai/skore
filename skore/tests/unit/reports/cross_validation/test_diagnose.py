from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skore import CrossValidationReport, configuration


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
    messages = report.diagnose()
    assert any(
        "[SKD002]" in message and "issue detected" in message for message in messages
    )


def test_diagnose_ignore(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    messages = report.diagnose(ignore=["SKD001"])
    assert all("[SKD001]" not in message for message in messages)


def test_diagnose_expensive_flag(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    assert len(report.diagnose(expensive=True)) == len(report.diagnose(expensive=False))


def test_diagnose_called_on_init(monkeypatch, binary_classification_data):
    calls = []

    def _diagnose(self, *, expensive=False, ignore=None):
        calls.append((expensive, ignore))
        return []

    monkeypatch.setattr(CrossValidationReport, "diagnose", _diagnose)
    X, y = binary_classification_data
    CrossValidationReport(LogisticRegression(), X, y, splitter=2, diagnose=True)
    assert calls == [(False, None)]


def test_diagnose_uses_global_ignore(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    assert any("[SKD001]" in message for message in report.diagnose())
    with configuration(ignore_diagnostics=["SKD001"]):
        assert all("[SKD001]" not in message for message in report.diagnose())


def test_diagnose_follows_global_config_default(binary_classification_data):
    X, y = binary_classification_data
    with configuration(diagnose=True):
        report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    assert hasattr(report, "_latest_diagnostics_")
    assert len(report._latest_diagnostics_) > 0
