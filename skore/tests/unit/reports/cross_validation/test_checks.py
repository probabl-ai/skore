import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from skrub import DatetimeEncoder, tabular_pipeline

from skore import Check, evaluate
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import (
    CheckHyperparamsAtSearchEdge,
    CheckSearchParamsToTune,
)


@pytest.fixture
def regression_report(regression_data):
    X, y = regression_data
    return evaluate(LinearRegression(), X, y, splitter=3)


class CVCheck(Check):
    code = "CVCUSTOM"
    title = "CV-level check"
    report_types = ["cross-validation"]
    docs_url = "cvcustom"

    def check_function(self, report):
        return f"Ran on {len(report.reports_)} splits."


class EstimatorCheck(Check):
    code = "ESTCUSTOM"
    title = "Estimator-level check"
    report_types = ["estimator"]
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
    """Check that the overfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    summary = report.metrics.summarize(data_source="test").summary
    n_metrics = len(
        {
            (row["metric_verbose_name"], row["label"], row["average"], row["output"])
            for row in summary.to_dict("records")
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
    """Check that the underfitting issue is detected."""
    X, y = regression_data
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(DummyRegressor(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    summary = report.metrics.summarize(data_source="test").summary
    n_metrics = len(
        {
            (row["metric_verbose_name"], row["label"], row["average"], row["output"])
            for row in summary.to_dict("records")
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
            report.metrics.summarize(data_source="test")._to_pivoted_frame(
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
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(LogisticRegression(), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD004" in issues.index
    assert "Accuracy should not be used alone" in issues.loc["SKD004", "explanation"]


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
def test_skd005_detects_underrepresented_classes(x_container, y_container):
    """Check that underrepresented classes are detected."""
    weights = [0.9, 0.05, 0.05]
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        n_clusters_per_class=1,
        weights=weights,
        random_state=0,
    )
    feature_columns = [str(i) for i in range(X.shape[1])]
    X = convert_container(X, x_container, column_names=feature_columns)
    y = convert_container(y, y_container)
    report = evaluate(LogisticRegression(max_iter=1000), X, y, splitter=3)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD005" in issues.index
    assert "Accuracy should not be used alone" in issues.loc["SKD005", "explanation"]


def test_skd006_detects_coefficient_interpretation(regression_data):
    """Check that the coefficient interpretation tip is emitted."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestRegressor(n_estimators=5, random_state=0),
        Pipeline([("rf", RandomForestRegressor(n_estimators=5, random_state=0))]),
    ],
)
def test_skd007_mdi_bias_with_high_cardinality(regression_data, estimator):
    """SKD007 tip is emitted with continuous features and tree importances."""
    X, y = regression_data
    report = evaluate(estimator, X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD007" in tips.index
    assert "High-cardinality features detected" in tips.loc["SKD007", "explanation"]


def test_skd008_correlated_features():
    """SKD008 issue is emitted when two features are near-perfectly correlated."""
    rng = np.random.RandomState(42)
    X = rng.standard_normal((100, 4))
    X[:, 1] = X[:, 0] + rng.standard_normal(100) * 1e-4
    y = rng.standard_normal(100)
    report = evaluate(
        LinearRegression(),
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
        splitter=3,
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD008" in issues.index
    assert "1 pair(s) of features" in issues.loc["SKD008", "explanation"]


def test_skd009_detects_worse_than_baseline(regression_data):
    """Check that the worse-than-baseline issue is detected."""
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


def test_skd010_detects_slower_than_baseline(regression_data, monkeypatch):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    from skore._sklearn._checks import model_checks
    from skore._sklearn._checks._utils import get_fitted_estimator

    def mock_get_fit_time(report):
        if isinstance(get_fitted_estimator(report), RandomForestRegressor):
            return 0.20
        return 0.05

    monkeypatch.setattr(model_checks, "get_fit_time", mock_get_fit_time)

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
    """Features correlated with the target get flagged as golden."""
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


def test_skd012_detects_useless_features():
    """Noise features are flagged when permutation importance is negligible."""
    X, y = make_regression(
        n_samples=300,
        n_features=6,
        n_informative=2,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )
    report = evaluate(Ridge(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD012" in tips.index
    assert "permutation importance" in tips.loc["SKD012", "explanation"]


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


def test_skd014_raises_at_numeric_edge_on_cv_report(regression_data, monkeypatch):
    """SKD014 flags when best hyperparameter is at the edge of the search space."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=2)
    report = evaluate(search, X, y, splitter=3)
    monkeypatch.setattr(report.reports_[0].estimator_, "best_params_", {"alpha": 0.1})
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    assert "minimum" in issues.loc["SKD014", "explanation"]


def test_skd014_not_applicable_for_plain_estimator_on_cv_report(regression_data):
    """SKD014 raises CheckNotApplicable with a plain estimator."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    with pytest.raises(CheckNotApplicable, match="not a BaseSearchCV"):
        CheckHyperparamsAtSearchEdge().check_function(report)


def test_skd015_suggests_missing_params_on_cv_report(regression_data):
    """SKD015 tip is emitted when the search grid misses recommended params."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"n_estimators": [10, 50]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    assert "max_features" in tips.loc["SKD015", "explanation"]


def test_skd015_not_applicable_plain_estimator_on_cv_report(regression_data):
    """SKD015 raises CheckNotApplicable with a plain estimator."""
    X, y = regression_data
    report = evaluate(Ridge(), X, y, splitter=3)
    with pytest.raises(CheckNotApplicable, match="not a BaseSearchCV"):
        CheckSearchParamsToTune().check_function(report)


def test_skd016_fires_on_default_estimator_on_cv_report(regression_data):
    """SKD016 fires when the estimator is left at sklearn defaults."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD016" in tips.index
    explanation = tips.loc["SKD016", "explanation"]
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation


def test_skd016_passed_when_tuned_on_cv_report(regression_data):
    """SKD016 passes once any recommended model param is set."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(max_depth=5), X, y, splitter=3)
    assert "SKD016" in set(report.checks.summarize().frame(section="passed")["code"])


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


def test_fast_mode_skips_slow_checks(regression_report):
    """fast_mode=True skips slow uncached checks."""
    codes = set(regression_report.checks.summarize(fast_mode=True).frame()["code"])
    slow_codes = {"SKD009", "SKD010", "SKD011", "SKD012"}
    assert slow_codes.isdisjoint(codes)
