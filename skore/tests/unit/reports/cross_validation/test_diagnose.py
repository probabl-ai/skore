import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from skore import Check, evaluate


@pytest.fixture
def regression_report(regression_data):
    X, y = regression_data
    return evaluate(LinearRegression(), X, y, splitter=3)


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


def test_skd003_detects_inconsistent_splits():
    """Check that the inconsistent performance across splits issue is detected."""
    X, y = make_classification(n_samples=400, n_features=5, random_state=0)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    assert "SKD003" not in set(report.diagnosis().frame(severity="issue")["code"])

    # Corrupt the first split
    y[0 : len(y) // 5] = np.random.RandomState(0).randint(0, 2, len(y) // 5)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    issues = report.diagnosis().frame(severity="issue").set_index("code")
    assert "SKD003" in issues.index
    assert "split #0" in issues.loc["SKD003", "explanation"]
    n_metrics = (
        len(
            report.metrics.summarize(data_source="test").frame(
                aggregate=None, flat_index=True
            )
        )
        - 2  # -2 for the timing metrics
    )
    assert f"for {n_metrics}/{n_metrics} metrics" in issues.loc["SKD003", "explanation"]


def test_skd001_aggregates_overfitting_across_splits(regression_data):
    """Check that the overfitting issue is aggregated across splits."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y, splitter=3)
    issues = report.diagnosis().frame(severity="issue").set_index("code")
    assert "SKD001" in issues.index
    assert "3/3 evaluated splits" in issues.loc["SKD001", "explanation"]


def test_skd002_aggregates_underfitting_across_splits(regression_data):
    """Check that the underfitting issue is aggregated across splits."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.diagnosis().frame(severity="issue").set_index("code")
    assert "SKD002" in issues.index
    assert "3/3 evaluated splits" in issues.loc["SKD002", "explanation"]


def test_reuses_split_cached_results(monkeypatch, regression_report):
    """Check that check results are cached and reused across splits."""
    regression_report.diagnosis()

    for sub_report in regression_report.estimator_reports_:
        for check in sub_report._checks_registry:
            monkeypatch.setattr(
                check,
                "check_function",
                lambda rpt: pytest.fail("re-ran cached check"),
            )

    regression_report.diagnosis()


def test_add_checks_cv_level(regression_report):
    """Check that add_checks registers a CV-level check."""
    regression_report.diagnosis.add([CVCheck()])
    issues = regression_report.diagnosis().frame(severity="issue").set_index("code")
    assert "CVCUSTOM" in issues.index
    assert issues.loc["CVCUSTOM", "title"] == "CV-level check"
    assert issues.loc["CVCUSTOM", "documentation_url"].endswith("#cvcustom")
    assert issues.loc["CVCUSTOM", "explanation"] == "Ran on 3 splits."


def test_add_checks_estimator_level(regression_report):
    """Check that add_checks with estimator report_type propagates and aggregates."""
    regression_report.diagnosis.add([EstimatorCheck()])
    issues = regression_report.diagnosis().frame(severity="issue").set_index("code")
    assert "ESTCUSTOM" in issues.index
    assert "3/3 evaluated splits" in issues.loc["ESTCUSTOM", "explanation"]
    assert "single split" not in issues.loc["ESTCUSTOM", "explanation"]
