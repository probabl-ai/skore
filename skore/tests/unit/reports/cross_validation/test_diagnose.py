from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skore import CrossValidationReport, EstimatorReport, configuration
from skore._sklearn._diagnostic import DiagnosticDisplay


def test_diagnose_aggregates_overfitting_across_splits():
    """Check that the overfitting issue is aggregated across splits."""
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
    result = report.diagnose()
    assert "SKD001" in result.issues
    assert "evaluated splits" in result.issues["SKD001"]["explanation"]


def test_diagnose_aggregates_underfitting_across_splits(binary_classification_data):
    """Check that the underfitting issue is aggregated across splits."""
    X, y = binary_classification_data
    report = CrossValidationReport(DummyClassifier(strategy="prior"), X, y, splitter=5)
    result = report.diagnose()
    assert "SKD002" in result.issues
    assert "evaluated splits" in result.issues["SKD002"]["explanation"]


def test_diagnose_ignore(binary_classification_data):
    """Check that checks are ignored when ignore is passed."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    result = report.diagnose(ignore=["SKD001"])
    assert "SKD001" not in result.issues


def test_diagnose_result_has_repr(binary_classification_data):
    """Check that the diagnostic result has a repr."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    results = report.diagnose()
    assert isinstance(results, DiagnosticDisplay)
    assert "Diagnostic:" in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle


def test_diagnose_reuses_split_cached_results(monkeypatch, binary_classification_data):
    """Check that check results are cached and reused across splits."""
    calls = 0
    original = EstimatorReport._run_checks

    def _run_checks_wrapper(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(EstimatorReport, "_run_checks", _run_checks_wrapper)
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    report.diagnose()
    calls_after_first = calls
    report.diagnose()
    assert calls_after_first == calls


def test_diagnose_uses_global_ignore(binary_classification_data):
    """Check that checks are ignored when global ignore is set."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=3)
    report.diagnose()
    _results, checked_codes = report._issues_cache
    assert len(checked_codes) > 0
    with configuration(ignore_checks=list(checked_codes)):
        assert report.diagnose().issues == {}
