import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skore import Check, evaluate


class CVCheck(Check):
    code = "CVCUSTOM"
    title = "CV-level check"
    report_type = "cross-validation"
    docs_url = "cvcustom"

    def check_function(self, report):
        return f"Ran on {len(report.estimator_reports_)} splits."


class EstimatorCheck(Check):
    code = "ESTCUSTOM"
    title = "Estimator-level check"
    report_type = "estimator"
    docs_url = "estcustom"

    def check_function(self, report):
        return "Detected on a single split."


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
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    report.diagnose()

    for sub_report in report.estimator_reports_:
        for check in sub_report._checks_registry:
            monkeypatch.setattr(
                check,
                "check_function",
                lambda rpt: pytest.fail("re-ran cached check"),
            )

    report.diagnose()


def test_add_checks_cv_level(regression_data):
    """Check that add_checks registers a CV-level check."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)

    report.add_checks([CVCheck()])
    result = report.diagnose()
    assert "CVCUSTOM" in result.issues
    assert result.issues["CVCUSTOM"]["title"] == "CV-level check"
    assert result.issues["CVCUSTOM"]["docs_url"] == "cvcustom"
    assert result.issues["CVCUSTOM"]["explanation"] == "Ran on 3 splits."


def test_add_checks_estimator_level(regression_data):
    """Check that add_checks with estimator report_type propagates and aggregates."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)

    report.add_checks([EstimatorCheck()])
    result = report.diagnose()
    assert "ESTCUSTOM" in result.issues
    assert "3/3 evaluated splits" in result.issues["ESTCUSTOM"]["explanation"]
    assert "single split" not in result.issues["ESTCUSTOM"]["explanation"]
