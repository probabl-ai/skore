import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression

from skore import Check, evaluate
from skore._externals._sklearn_compat import convert_container


@pytest.fixture
def regression_report(regression_data):
    X, y = regression_data
    return evaluate(LinearRegression(), X, y, splitter=3)


class CVCheck(Check):
    code = "CVCUSTOM"
    title = "CV-level check"
    report_type = ["cross-validation"]
    docs_url = "cvcustom"

    def check_function(self, report):
        return f"Ran on {len(report.reports_)} splits."


class EstimatorCheck(Check):
    code = "ESTCUSTOM"
    title = "Estimator-level check"
    report_type = ["estimator"]
    docs_url = "estcustom"

    def check_function(self, report):
        return "Detected on a single split."


def test_cv_passed_checks_appear_in_summarize(regression_data):
    """Passed CV-scoped checks without findings appear on the CV report summary."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    passed_codes = set(report.checks.summarize().frame(section="passed")["code"])
    assert "SKD003" in passed_codes


def test_skd003_detects_inconsistent_splits():
    """Check that the inconsistent performance across splits issue is detected."""
    X, y = make_classification(n_samples=400, n_features=5, random_state=0)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    assert "SKD003" not in set(report.checks.summarize().frame(section="issue")["code"])

    # Corrupt the first split
    y[0 : len(y) // 5] = np.random.RandomState(0).randint(0, 2, len(y) // 5)
    report = evaluate(LogisticRegression(random_state=0), X, y, splitter=5)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
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


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
def test_skd004_detects_high_class_imbalance(x_container, y_container):
    """Check that high class imbalance is detected with several container types."""
    weights = [0.9, 0.1]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD004" in issues.index
    assert "Accuracy should not be used alone" in issues.loc["SKD004", "explanation"]


def test_reuses_cv_cached_results(monkeypatch, regression_report):
    """Check that CV-level check results are cached and reused."""
    regression_report.checks.summarize()

    for check in regression_report._checks_registry:
        if check.code == "SKD003":
            monkeypatch.setattr(
                check,
                "check_function",
                lambda rpt: pytest.fail("re-ran cached check"),
            )

    regression_report.checks.summarize()


def test_add_checks_cv_level(regression_report):
    """Check that add_checks registers a CV-level check."""
    regression_report.checks.add([CVCheck()])
    issues = (
        regression_report.checks.summarize().frame(section="issue").set_index("code")
    )
    assert "CVCUSTOM" in issues.index
    assert issues.loc["CVCUSTOM", "title"] == "CV-level check"
    assert issues.loc["CVCUSTOM", "documentation_url"].endswith("#cvcustom")
    assert issues.loc["CVCUSTOM", "explanation"] == "Ran on 3 splits."


def test_add_checks_estimator_level_not_on_cv_summary(regression_report):
    """Estimator-scoped custom checks do not run on the CV report summary."""
    regression_report.checks.add([EstimatorCheck()])
    summary = regression_report.checks.summarize().frame()
    assert "ESTCUSTOM" not in set(summary["code"])
