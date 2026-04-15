from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skore import Check, evaluate


def test_diagnose_aggregates_overfitting_across_splits(regression_data):
    """Check that the overfitting issue is aggregated across splits."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y, splitter=3)
    result = report.diagnose()
    assert "SKD001" in result.issues
    assert "3/3 evaluated splits" in result.issues["SKD001"]["explanation"]


def test_diagnose_aggregates_underfitting_across_splits(regression_data):
    """Check that the underfitting issue is aggregated across splits."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    result = report.diagnose()
    assert "SKD002" in result.issues
    assert "3/3 evaluated splits" in result.issues["SKD002"]["explanation"]


def test_diagnose_reuses_split_cached_results(monkeypatch, regression_data):
    """Check that check results are cached and reused across splits."""
    calls = 0
    original_run = Check.run

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "run", counting_run)
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    report.diagnose()
    calls_after_first = calls
    report.diagnose()
    assert calls_after_first == calls


def test_add_checks_cv_level(regression_data):
    """Check that add_checks registers a CV-level check."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)

    def cv_check(report):
        n_splits = len(report.estimator_reports_)
        return f"Ran on {n_splits} splits."

    check = Check(
        cv_check, "CVCUSTOM", "CV-level check", "cross-validation", "cvcustom"
    )
    report.add_checks([check])
    result = report.diagnose()
    assert "CVCUSTOM" in result.issues
    assert result.issues["CVCUSTOM"]["title"] == "CV-level check"
    assert result.issues["CVCUSTOM"]["docs_url"] == "cvcustom"
    assert result.issues["CVCUSTOM"]["explanation"] == "Ran on 3 splits."


def test_add_checks_estimator_level(regression_data):
    """Check that add_checks with estimator report_type propagates and aggregates."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)

    def estimator_check(report):
        return "Detected on a single split."

    check = Check(
        estimator_check,
        "ESTCUSTOM",
        "Estimator-level check",
        "estimator",
        "estcustom",
    )
    report.add_checks([check])
    result = report.diagnose()
    assert "ESTCUSTOM" in result.issues
    assert "3/3 evaluated splits" in result.issues["ESTCUSTOM"]["explanation"]
    assert "single split" not in result.issues["ESTCUSTOM"]["explanation"]
