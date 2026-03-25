from unittest.mock import patch

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skore import CrossValidationReport, EstimatorReport, configuration
from skore._sklearn._diagnostics.base import DiagnosticResults


def test_diagnose_aggregates_overfitting_across_splits():
    """Check the overfitting diagnostics are aggregated across splits."""
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
    assert any("[SKD001]" in message for message in messages)
    assert any("evaluated splits" in message for message in messages)


def test_diagnose_aggregates_underfitting_across_splits(binary_classification_data):
    """Check the underfitting diagnostics are aggregated across splits."""
    X, y = binary_classification_data
    report = CrossValidationReport(DummyClassifier(strategy="prior"), X, y, splitter=5)
    results = report.diagnose()
    assert any("[SKD002]" in message for message in results)
    assert any("evaluated splits" in message for message in results)


def test_diagnose_ignore(binary_classification_data):
    """Check the diagnostics are ignored when ignore is passed."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    messages = report.diagnose(ignore=["SKD001"])
    assert all("[SKD001]" not in message for message in messages)


def test_diagnose_called_on_init(monkeypatch, binary_classification_data):
    """Check the diagnostics are called on init."""
    calls = []

    def diagnose(self):
        calls.append(True)
        return [], set()

    monkeypatch.setattr(CrossValidationReport, "diagnose", diagnose)
    X, y = binary_classification_data
    CrossValidationReport(LogisticRegression(), X, y, splitter=2, diagnose=True)
    assert calls == [True]


def test_diagnose_result_has_repr(binary_classification_data):
    """Check the diagnostics result has a repr."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    results = report.diagnose()
    assert isinstance(results, DiagnosticResults)
    assert "Diagnostics:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle


def test_diagnose_reuses_split_cached_diagnostics(
    monkeypatch, binary_classification_data
):
    """Check the diagnostics are reused across splits."""
    calls = 0
    original = EstimatorReport._compute_diagnostics

    def _compute_diagnostics(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(EstimatorReport, "_compute_diagnostics", _compute_diagnostics)
    X, y = binary_classification_data
    report = CrossValidationReport(
        LogisticRegression(), X, y, splitter=3, diagnose=True
    )
    calls_after_init = calls
    report.diagnose()
    assert calls_after_init == calls


def test_diagnose_uses_global_ignore(binary_classification_data):
    """Check the diagnostics are ignored when global ignore is set."""
    X, y = binary_classification_data
    report = CrossValidationReport(
        LogisticRegression(), X, y, splitter=3, diagnose=True
    )
    _results, checked_codes = report._diagnostics_cache
    assert len(checked_codes) > 0
    with configuration(ignore_diagnostics=list(checked_codes)):
        assert report.diagnose().diagnostics == []


def test_diagnose_follows_global_config_default(binary_classification_data):
    """Check the diagnostics are displayed when global diagnose is set."""
    X, y = binary_classification_data
    with patch.object(
        CrossValidationReport, "_display_diagnose_results"
    ) as display_mock:
        CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    display_mock.assert_not_called()
    with (
        patch.object(
            CrossValidationReport, "_display_diagnose_results"
        ) as display_mock,
        configuration(diagnose=True),
    ):
        CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    display_mock.assert_called_once()
