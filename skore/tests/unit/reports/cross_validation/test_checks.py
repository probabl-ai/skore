import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from skrub import DatetimeEncoder, tabular_pipeline

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


def test_skd001_detects_overfitting(regression_data):
    """Check that the overfitting issue is detected on a CV report."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = len(
        {
            (row["metric_verbose_name"], row["label"], row["average"], row["output"])
            for row in report.metrics.summarize(data_source="test").rows
            if row["metric_verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
        }
    )
    assert "SKD001" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} default predictive metrics"
        in issues.loc["SKD001", "explanation"]
    )


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
def test_skd002_detects_underfitting(regression_data, x_container, y_container):
    """Check that the underfitting issue is detected on a CV report."""
    X, y = regression_data
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = len(
        {
            (row["metric_verbose_name"], row["label"], row["average"], row["output"])
            for row in report.metrics.summarize(data_source="test").rows
            if row["metric_verbose_name"] not in {"Fit time (s)", "Predict time (s)"}
        }
    )
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


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


def test_skd009_detects_worse_than_baseline(regression_data):
    """Check that the worse-than-baseline issue is detected on a CV report."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index
    assert (
        "not significantly better than a HistGradientBoosting baseline"
        in issues.loc["SKD009", "explanation"]
    )


def test_skd009_not_detected_on_strong_model():
    """Check that SKD009 is not detected when the model beats HistGradientBoosting."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(RidgeCV(), X, y, splitter=3)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD009" not in codes


def test_skd010_detects_slower_than_baseline(regression_data):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    X, y = regression_data
    report = evaluate(
        RandomForestRegressor(n_estimators=200, random_state=0), X, y, splitter=3
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD010" in issues.index
    assert "slower than a fast linear baseline" in issues.loc["SKD010", "explanation"]


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), tabular_pipeline(LinearRegression())]
)
def test_skd011_detects_golden_feature(estimator):
    """Features correlated with the target get flagged as golden on a CV report."""
    rng = np.random.RandomState(0)
    n_samples = 200
    X = rng.normal(size=(n_samples, 4))
    y = X[:, 0] * 10
    X[:, 1] = y + rng.normal(scale=0.01, size=n_samples)
    report = evaluate(
        estimator,
        pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])]),
        pd.Series(y),
        splitter=3,
    )
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD011" in tips.index
    explanation = tips.loc["SKD011", "explanation"]
    assert "Feature 0" in explanation
    assert "Feature 1" in explanation
    assert "Feature 2" not in explanation
    assert "Feature 3" not in explanation


def test_skd013_train_test_time_overlap():
    """Shuffled CV triggers overlap; time-series CV passes."""
    n = 200
    X = pd.DataFrame(
        {
            "feat": np.arange(n, dtype=float),
            "date": pd.date_range("2026-12-01", periods=n, freq="D"),
        }
    )
    y = np.arange(n, dtype=float)
    pipe = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("date", DatetimeEncoder(), ["date"])],
                    remainder="passthrough",
                ),
            ),
            ("reg", LinearRegression()),
        ]
    )

    report = evaluate(
        pipe, X, y, splitter=KFold(n_splits=5, shuffle=True, random_state=0)
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD013" in issues.index
    assert "date" in issues.loc["SKD013", "explanation"]

    report = evaluate(pipe, X, y, splitter=TimeSeriesSplit(n_splits=5))
    summary = report.checks.summarize()
    assert "SKD013" not in set(summary.frame(section="issue")["code"])
    assert "SKD013" in set(summary.frame(section="passed")["code"])


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
