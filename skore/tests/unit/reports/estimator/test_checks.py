from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import (
    BayesianRidge,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skrub import DatetimeEncoder, tabular_pipeline

from skore import Check, EstimatorReport, configuration, evaluate
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.base import (
    ChecksSummaryDisplay,
    _get_issue_documentation_url,
)
from skore._sklearn._checks.model_checks import (
    CheckEstimatorNotTuned,
    CheckHyperparamsAtSearchEdge,
    CheckSearchParamsToTune,
)
from skore._utils._testing import MockEstimator


@pytest.fixture(params=[LinearRegression(), tabular_pipeline(LinearRegression())])
def regression_report(request, regression_data):
    X, y = regression_data
    return evaluate(
        request.param,
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
    )


def mock_issue(report, ignored_codes, *, fast_mode=False):
    return (
        {
            "SKD001": {
                "title": "Mock title",
                "docs_url": "skd001-overfitting",
                "explanation": "Mock overfitting detected.",
                "severity": "issue",
            }
        },
        {"SKD001"},
        set(),
    )


class MockCheck(Check):
    code = "TST001"
    title = "Test issue"
    report_type = "estimator"
    docs_url = "tst001"

    def __init__(
        self, has_issue: bool = True, docs_url="tst001", report_type="estimator"
    ):
        self.has_issue = has_issue
        self.docs_url = docs_url
        self.report_type = report_type

    def check_function(self, report):
        return "Something was found." if self.has_issue else None


def test_skd001_detects_overfitting(regression_data):
    """Check that the overfitting issue is detected."""
    X, y = regression_data
    report = evaluate(DecisionTreeRegressor(random_state=0), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD001" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} default predictive metrics"
        in issues.loc["SKD001", "explanation"]
    )


@pytest.mark.parametrize(
    "x_container, y_container",
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
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.parametrize(
    "weights, code", [([0.9, 0.1], "SKD004"), ([0.9, 0.05, 0.05], "SKD005")]
)
def test_skd004_skd005_detects_high_class_imbalance(
    weights, code, x_container, y_container
):
    """Check that the high class imbalance issue is detected."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=3,
        n_classes=len(weights),
        random_state=0,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    result = report.checks.summarize()
    assert code not in set(result.frame(section="issue")["code"])

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
    assert code in issues.index
    assert "Accuracy should not be used alone" in issues.loc[code, "explanation"]


def test_skd006_detects_coefficient_interpretation(regression_data):
    """Check that the coefficient interpretation tip is emitted."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


@pytest.mark.parametrize(
    "pipeline, expected_message",
    [
        (
            Pipeline([("model", LinearRegression())]),
            "Features are not on the same scale",
        ),
        (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "Features appear to be standardized",
        ),
    ],
)
def test_skd006_pipeline_coefficient_interpretation(
    regression_data, pipeline, expected_message
):
    """SKD006 tip reflects preprocessed feature scale in a pipeline."""
    X, y = regression_data
    report = evaluate(pipeline, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert expected_message in tips.loc["SKD006", "explanation"]


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
    report = evaluate(estimator, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD007" in tips.index
    assert (
        "High-cardinality features detected: Feature 0, Feature 1, Feature 2 "
        "(and 1 more)" in tips.loc["SKD007", "explanation"]
    )


def test_skd007_not_emitted_for_binary_features():
    """SKD007 tip is absent when all features are low-cardinality."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 2, size=(20, 4)).astype(float)
    y = rng.standard_normal(20)
    report = evaluate(RandomForestRegressor(n_estimators=5, random_state=0), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD007" not in tips.index


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), tabular_pipeline(LinearRegression())]
)
def test_skd008_correlated_features(estimator):
    """SKD008 issue is emitted when two features are near-perfectly correlated."""
    rng = np.random.RandomState(42)
    X = rng.standard_normal((20, 4))
    X[:, 1] = X[:, 0] + rng.standard_normal(20) * 1e-4
    y = rng.standard_normal(20)
    report = evaluate(
        estimator,
        pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]),
        pd.Series(y),
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD008" in issues.index
    assert "1 pair(s) of features" in issues.loc["SKD008", "explanation"]


def test_skd008_not_emitted_for_independent_features(regression_data):
    """SKD008 issue is absent when features are independent."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD008" not in issues.index


def test_skd009_detects_worse_than_baseline(regression_data):
    """Check that the worse-than-baseline issue is detected on a dummy estimator."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD009" in issues.index
    assert (
        "not significantly better than a HistGradientBoosting baseline"
        in issues.loc["SKD009", "explanation"]
    )


def test_skd009_not_detected_on_strong_model(regression_data):
    """Check that SKD009 is not detected when the model beats HistGradientBoosting."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD009" not in codes


def test_skd010_detects_slower_than_baseline(regression_data):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(n_estimators=200, random_state=0), X, y)
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
        splitter=0.2,
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
    report = evaluate(Ridge(), X, y, splitter=0.2)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD012" in tips.index
    explanation = tips.loc["SKD012", "explanation"]
    assert "permutation importance" in explanation
    assert "Feature #0" not in explanation
    assert "Feature #1" not in explanation
    assert "Feature #2" in explanation
    assert "Feature #3" in explanation
    assert "Feature #4" in explanation
    assert "Feature #5" in explanation


def test_skd013_train_test_time_overlap():
    """Shuffled split triggers overlap; proper temporal split passes."""
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )
    report = EstimatorReport(
        pipe, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD013" in issues.index
    assert "date" in issues.loc["SKD013", "explanation"]

    split = int(n * 0.8)
    report = EstimatorReport(
        pipe,
        X_train=X.iloc[:split],
        y_train=y[:split],
        X_test=X.iloc[split:],
        y_test=y[split:],
    )
    summary = report.checks.summarize()
    assert "SKD013" not in set(summary.frame(section="issue")["code"])
    assert "SKD013" in set(summary.frame(section="passed")["code"])


def test_skd010_not_detected_for_fast_model(regression_data):
    """Check that SKD010 does not fire when the model is not slower than baseline."""
    X, y = regression_data
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD010" not in codes


@pytest.mark.parametrize(
    "estimator, param_name, side, expected",
    [
        # Ridge.alpha: Interval(Real, 0, None, closed='left') -> left bound is 0.0
        (Ridge(), "alpha", "left", 0.0),
        # Ridge.alpha has no finite right bound
        (Ridge(), "alpha", "right", None),
        # BayesianRidge.tol: Interval(Real, 0, None, closed='neither') -> open, no bound
        (BayesianRidge(), "tol", "left", None),
        # estimator without _parameter_constraints
        (MockEstimator(error=ValueError("unused")), "alpha", "left", None),
        # Pipeline: navigate 'ridge__alpha' to Ridge.alpha left bound
        (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            "ridge__alpha",
            "left",
            0.0,
        ),
    ],
)
def test_get_space_bound(estimator, param_name, side, expected):
    assert (
        CheckHyperparamsAtSearchEdge._get_space_bound(
            estimator, param_name=param_name, side=side
        )
        == expected
    )


def _prefit_grid_search_report(X, y, search):
    search.fit(X, y)
    return evaluate(search, X, y, splitter="prefit")


@pytest.mark.parametrize(
    "param_grid", [{"alpha": [0.1, 1.0, 10.0]}, {"alpha": [10.0, 0.1, 1.0]}]
)
def test_skd014_raises_at_numeric_edge(regression_data, monkeypatch, param_grid):
    """SKD014 flags when best is at the numeric min/max of tried values."""
    X, y = regression_data
    report = _prefit_grid_search_report(
        X, y, GridSearchCV(Ridge(), param_grid=param_grid, cv=2)
    )
    monkeypatch.setattr(report.estimator_, "best_params_", {"alpha": 0.1})
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    explanation = issues.loc["SKD014", "explanation"]
    assert "alpha" in explanation
    assert "minimum" in explanation


@pytest.mark.parametrize(
    "param_grid", [{"alpha": [1.0, 2.0, 3.0]}, {"alpha": [3.0, 1.0, 2.0]}]
)
def test_skd014_not_raised_for_interior_best(regression_data, monkeypatch, param_grid):
    """SKD014 is absent when the best value is not at the tried min or max."""
    X, y = regression_data
    report = _prefit_grid_search_report(
        X, y, GridSearchCV(Ridge(), param_grid=param_grid, cv=2)
    )
    monkeypatch.setattr(report.estimator_, "best_params_", {"alpha": 2.0})
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


@pytest.mark.parametrize("prefit", [True, False])
def test_skd014_prefit_and_evaluate_fit(regression_data, monkeypatch, prefit):
    """SKD014 runs for prefit and evaluate-fitted GridSearchCV reports."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid={"alpha": [10.0, 0.1, 1.0]}, cv=2)
    if prefit:
        report = _prefit_grid_search_report(X, y, search)
    else:
        report = evaluate(search, X, y)
    monkeypatch.setattr(report.estimator_, "best_params_", {"alpha": 0.1})
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    assert "minimum" in issues.loc["SKD014", "explanation"]


def test_skd014_not_applicable_for_plain_estimator(regression_data):
    """SKD014 raises CheckNotApplicable when the report estimator isn't BaseSearchCV."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    with pytest.raises(CheckNotApplicable):
        CheckHyperparamsAtSearchEdge().check_function(report)


@pytest.mark.parametrize(
    "param_grid", [{"fit_intercept": [False, True]}, {"solver": ["svd", "cholesky"]}]
)
def test_skd014_skips_non_numeric_hyperparameters(regression_data, param_grid):
    """SKD014 ignores bool, string, and other non-numeric search parameters."""
    X, y = regression_data
    report = _prefit_grid_search_report(
        X, y, GridSearchCV(Ridge(), param_grid=param_grid, cv=2)
    )
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


@pytest.mark.parametrize(
    "search",
    [
        GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=2),
        RandomizedSearchCV(
            Ridge(), param_distributions={"alpha": [0.1, 1.0, 10.0]}, cv=2
        ),
    ],
)
def test_skd014_search_classes(regression_data, monkeypatch, search):
    """SKD014 runs for GridSearchCV and RandomizedSearchCV using cv_results_."""
    X, y = regression_data
    report = _prefit_grid_search_report(X, y, search)
    monkeypatch.setattr(report.estimator_, "best_params_", {"alpha": 0.1})
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    assert "minimum" in issues.loc["SKD014", "explanation"]


@pytest.mark.parametrize(
    "search, best_params",
    [
        (
            GridSearchCV(Ridge(), param_grid={"alpha": [0.0, 1.0, 10.0]}, cv=2),
            {"alpha": 0.0},
        ),
        (
            GridSearchCV(
                Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
                param_grid={"ridge__alpha": [0.0, 1.0, 10.0]},
                cv=2,
            ),
            {"ridge__alpha": 0.0},
        ),
    ],
)
def test_skd014_not_raised_when_search_edge_matches_space_edge(
    regression_data, monkeypatch, search, best_params
):
    """SKD014 is absent when the search minimum equals the parameter-space minimum."""
    X, y = regression_data
    report = _prefit_grid_search_report(X, y, search)
    monkeypatch.setattr(report.estimator_, "best_params_", best_params)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


def test_skd015_suggests_missing_params(regression_data):
    """SKD015 tip is emitted when the search grid misses recommended params."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"n_estimators": [10, 50]},
        cv=2,
    )
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "max_features" in explanation
    assert "min_samples_leaf" in explanation
    assert "max_depth" not in explanation
    assert "n_estimators" not in explanation


def test_skd015_passes_when_all_recommended_covered(regression_data):
    """SKD015 passes when every recommended param is already searched."""
    X, y = regression_data
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=2,
    )
    report = evaluate(search, X, y)
    summary = report.checks.summarize()
    assert "SKD015" in set(summary.frame(section="passed")["code"])


def test_skd015_not_applicable_plain_estimator(regression_data):
    """SKD015 raises CheckNotApplicable on a plain (non-search) estimator."""
    X, y = regression_data
    report = evaluate(Ridge(), X, y)
    with pytest.raises(CheckNotApplicable):
        CheckSearchParamsToTune().check_function(report)


def test_skd015_not_applicable_unknown_estimator(regression_data):
    """SKD015 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(),
        param_grid={"strategy": ["mean", "median"]},
        cv=2,
    )
    report = evaluate(search, X, y)
    with pytest.raises(CheckNotApplicable):
        CheckSearchParamsToTune().check_function(report)


def test_skd015_pipeline_single_step(regression_data):
    """SKD015 strips pipeline prefixes correctly for a single tuned step."""
    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    search = GridSearchCV(pipe, param_grid={"rf__n_estimators": [10, 50]}, cv=2)
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation
    assert "n_estimators" not in explanation


def test_skd015_pipeline_multi_step(binary_classification_data):
    """SKD015 reports missing params for multiple pipeline steps."""
    X, y = binary_classification_data
    pipe = Pipeline(
        [
            ("pca", PCA()),
            ("rbf", RBFSampler(n_components=2)),
            ("clf", RandomForestClassifier(random_state=0, n_estimators=10)),
        ]
    )
    search = GridSearchCV(
        pipe,
        param_grid={
            "clf__min_samples_leaf": [10, 50],
            "rbf__n_components": [2, 3],
        },
        cv=2,
    )
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "StandardScaler" not in explanation
    assert "PCA" in explanation
    assert "n_components" in explanation
    assert "RBFSampler" in explanation
    assert "gamma" in explanation
    assert "RandomForestClassifier" in explanation
    assert "max_features" in explanation


def test_skd015_pipeline_flags_untuned_step(regression_data):
    """SKD015 flags pipeline steps whose params are not in the grid at all."""
    X, y = regression_data
    pipe = Pipeline([("pca", PCA()), ("ridge", Ridge())])
    search = GridSearchCV(pipe, param_grid={"ridge__alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "PCA" in explanation
    assert "n_components" in explanation


def test_skd015_equivalent_params_not_suggested(regression_data):
    """Tuning max_depth should not suggest min_samples_leaf or min_samples_split."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"max_depth": [3, 5, 10]},
        cv=2,
    )
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "min_samples_leaf" not in explanation
    assert "min_samples_split" not in explanation
    assert "max_features" in explanation


def test_skd016_fires_on_default_estimator(regression_data):
    """SKD016 fires when the estimator is left at sklearn defaults."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD016" in tips.index
    explanation = tips.loc["SKD016", "explanation"]
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation
    assert "min_samples_leaf" in explanation


def test_skd016_passed_when_tuned(regression_data):
    """SKD016 passes once any recommended-or-other model param is set."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(max_depth=5), X, y)
    assert "SKD016" in set(report.checks.summarize().frame(section="passed")["code"])


def test_skd016_ignores_infrastructure(regression_data):
    """Setting only infrastructure params (random_state) still triggers SKD016."""
    X, y = regression_data
    report = evaluate(Ridge(random_state=42), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD016" in tips.index
    assert "alpha" in tips.loc["SKD016", "explanation"]


def test_skd016_ignores_budget_params(regression_data):
    """Raising max_iter alone still triggers SKD016."""
    X, y = regression_data
    report = evaluate(Ridge(max_iter=200), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD016" in tips.index
    assert "alpha" in tips.loc["SKD016", "explanation"]


def test_skd016_not_applicable_unknown_estimator(regression_data):
    """SKD016 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    report = evaluate(DummyRegressor(), X, y)
    with pytest.raises(CheckNotApplicable):
        CheckEstimatorNotTuned().check_function(report)


def test_skd016_not_applicable_search(regression_data):
    """SKD016 defers to SKD015 when the estimator is a search."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y)
    with pytest.raises(CheckNotApplicable):
        CheckEstimatorNotTuned().check_function(report)


def test_skd016_pipeline_walks_steps(regression_data):
    """SKD016 reports only the pipeline steps that are still at defaults."""
    X, y = regression_data
    X, y = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]), pd.Series(y)
    pipe = Pipeline([("pca", PCA()), ("ridge", Ridge(alpha=2.0))])
    report = evaluate(pipe, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD016" in tips.index
    explanation = tips.loc["SKD016", "explanation"]
    assert "PCA" in explanation
    assert "n_components" in explanation
    assert "Ridge" not in explanation


def test_ignore_checks(monkeypatch, regression_report):
    """Check that checks are ignored when ignore is passed."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert (
        regression_report.checks.summarize(ignore=["SKD001"])
        .frame(section="issue")
        .empty
    )


def test_exception_when_train_data_missing(regression_train_test_split):
    """Check that an exception is raised when the train data is missing."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    for check in report._checks_registry:
        if check.code in ["SKD001", "SKD002", "SKD009", "SKD010"]:
            with pytest.raises(CheckNotApplicable):
                check.check_function(report)


def test_not_applicable_reason_in_summarize(regression_train_test_split):
    """Not-applicable checks surface their reason in the summary."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    na = report.checks.summarize().frame(section="not_applicable").set_index("code")
    assert "SKD001" in na.index
    assert na.loc["SKD001", "explanation"] == ("Train data is unavailable.")


def test_exception_when_baseline_report_creation_fails(regression_data, monkeypatch):
    """Check that an exception is raised when the baseline report creation fails."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    for check in report._checks_registry:
        if check.code in ["SKD002", "SKD009", "SKD010"]:
            with pytest.raises(CheckNotApplicable):
                check.check_function(report)


def test_no_issues(monkeypatch, regression_report):
    """Check that no issues are detected when checks pass."""
    monkeypatch.setattr(
        EstimatorReport,
        "_get_results",
        lambda report, ignored_codes, *, fast_mode=False: (
            {},
            {"SKD001", "SKD002"},
            set(),
        ),
    )
    assert regression_report.checks.summarize().frame(section="issue").empty


def test_checks_summary_repr(monkeypatch, regression_report):
    """Check that the checks summary has a repr."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    results = regression_report.checks.summarize()
    assert isinstance(results, ChecksSummaryDisplay)
    elements = [
        "Checks summary:",
        "Mock title.",
        "[SKD001]",
        "Mock overfitting detected",
    ]
    for element in elements:
        assert element in repr(results)
    bundle = results._repr_mimebundle_()
    assert "text/plain" in bundle
    assert "text/html" in bundle
    assert 'href="' in bundle["text/html"]
    assert "user_guide/automated_checks.html#" in bundle["text/html"]


def test_global_ignore(monkeypatch, regression_report):
    """Check that checks are ignored when global ignore is set."""
    monkeypatch.setattr(EstimatorReport, "_get_results", mock_issue)
    assert "SKD001" in set(
        regression_report.checks.summarize().frame(section="issue")["code"]
    )
    with configuration(ignore_checks=["SKD001"]):
        assert "SKD001" not in set(
            regression_report.checks.summarize().frame(section="issue")["code"]
        )


def test_documentation_url_points_to_existing_rst():
    """Check that the URL in _get_issue_documentation_url maps to a real RST file."""
    url = urlparse(_get_issue_documentation_url(mock_issue(None, set())[0]["SKD001"]))
    # url.path is e.g. "/dev/user_guide/automated_checks.html"
    # strip version prefix and convert .html -> .rst
    rst_rel_path = "/".join(url.path.split("/")[2:]).replace(".html", ".rst")
    rst_path = Path(__file__).parents[5] / "sphinx" / rst_rel_path
    assert rst_path.is_file()


def test_reuses_cached_results(monkeypatch, regression_report):
    """Check that check results are cached and reused."""
    calls = 0
    original_run = Check.check_function

    def counting_run(self, report):
        nonlocal calls
        calls += 1
        return original_run(self, report)

    monkeypatch.setattr(Check, "check_function", counting_run)
    regression_report.checks.summarize()
    calls_after_first = calls
    regression_report.checks.summarize()
    assert calls == calls_after_first


def test_add_checks_runs_custom_check(regression_report):
    """Check that add_checks runs the custom check and includes its issue."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    issues = (
        regression_report.checks.summarize().frame(section="issue").set_index("code")
    )
    assert "TST001" in issues.index
    assert issues.loc["TST001", "title"] == "Test issue"
    assert issues.loc["TST001", "documentation_url"].endswith("#tst001")
    assert issues.loc["TST001", "explanation"] == "Something was found."


def test_add_checks_reuses_builtin_cache(monkeypatch, regression_report):
    """Check that add_checks does not re-run already cached built-in checks."""
    regression_report.checks.summarize()

    for check in regression_report._checks_registry:
        monkeypatch.setattr(
            check, "check_function", lambda report: pytest.fail("re-ran cached check")
        )

    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()


def test_add_checks_docs_url_full(regression_report):
    """Check that a full https docs_url is preserved as-is in frame()."""
    check = MockCheck(has_issue=True, docs_url="https://example.com/my-doc")
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame.query("code == 'TST001'")
    assert row["documentation_url"].iloc[0] == "https://example.com/my-doc"
    assert "Read more about this here" in repr(result)


def test_add_checks_docs_url_absent(regression_report):
    """Check that missing docs_url results in None in the frame."""
    check = MockCheck(has_issue=True, docs_url=None)
    regression_report.checks.add([check])
    result = regression_report.checks.summarize()
    frame = result.frame()
    row = frame[frame["code"] == "TST001"]
    assert row["documentation_url"].isna().all()


def test_available_returns_code_dash_title(regression_report):
    """Check that available returns strings in 'code - title' format."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    available = regression_report.checks.available()
    assert "TST001 - Test issue" in available


def test_remove_checks_excludes_results(regression_report):
    """Check that remove excludes checks from results and available checks."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001" in set(regression_report.checks.summarize().frame()["code"])

    regression_report.checks.remove("TST001")
    assert "TST001" not in set(regression_report.checks.summarize().frame()["code"])
    assert "TST001 - Test issue" not in regression_report.checks.available()


def test_remove_clears_cache(regression_report):
    """Check that remove invalidates cached results for the removed check."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    regression_report.checks.summarize()
    assert "TST001" in regression_report._check_results_cache
    assert "TST001" in regression_report._applicable_codes

    regression_report.checks.remove("TST001")
    assert "TST001" not in regression_report._check_results_cache
    assert "TST001" not in regression_report._applicable_codes
    assert "TST001" not in regression_report._not_applicable_codes


def test_remove_is_case_insensitive(regression_report):
    """Check that remove matches check codes case-insensitively."""
    regression_report.checks.add([MockCheck(has_issue=True)])
    assert "TST001 - Test issue" in regression_report.checks.available()
    regression_report.checks.remove("tst001")
    assert "TST001 - Test issue" not in regression_report.checks.available()


def test_check_invalid_report_type(regression_report):
    """Check that Check raises ValueError for unsupported report_type."""
    check = MockCheck(has_issue=False, report_type="invalid")
    with pytest.raises(ValueError, match="report_type should be one of"):
        regression_report.checks.add([check])


def test_check_invalid_protocol(regression_report):
    """Check that Check raises ValueError for unsupported protocol."""

    class InvalidCheck:
        code = "INVALID001"
        title = "Invalid issue"
        report_type = "estimator"
        docs_url = "invalid001"

    with pytest.raises(ValueError, match="does not implement the Check protocol."):
        regression_report.checks.add([InvalidCheck()])


def test_custom_metric(binary_classification_data):
    """Check that checks works with custom metrics in the report."""
    X, y = binary_classification_data
    report = evaluate(DummyClassifier(), X, y, pos_label=1)
    report.metrics.add("f1")
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    n_metrics = report.metrics.summarize(data_source="test").data.shape[0] - 2
    assert "SKD002" in issues.index
    assert (
        f"for {n_metrics}/{n_metrics} comparable metrics"
        in issues.loc["SKD002", "explanation"]
    )


class TipCheck(Check):
    code = "TST002"
    title = "Tip check"
    report_type = "estimator"
    docs_url = "tst_tip"
    severity = "tip"

    def check_function(self, report):
        return "Be careful about this."


def test_tip_goes_to_tips_not_issues(regression_report):
    """A check with section='tip' is routed to tips, not issues."""
    regression_report.checks.add([TipCheck()])
    result = regression_report.checks.summarize()
    tips = result.frame(section="tip").set_index("code")
    assert "TST002" in tips.index
    assert "TST002" not in set(result.frame(section="issue")["code"])
    assert tips.loc["TST002", "section"] == "tip"


def test_passed_contains_applicable_checks_with_no_finding(regression_report):
    """Checks that ran without reporting anything show up as passed."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize()
    assert "TST001" in set(result.frame(section="passed")["code"])
    assert "TST001" not in set(result.frame(section="issue")["code"])
    assert "TST001" not in set(result.frame(section="tip")["code"])


def test_passed_excludes_ignored(regression_report):
    """Ignored codes are not listed as passed."""
    regression_report.checks.add([MockCheck(has_issue=False)])
    result = regression_report.checks.summarize(ignore=["TST001"])
    assert "TST001" not in set(result.frame(section="passed")["code"])
    assert "TST001" not in set(result.frame(section="issue")["code"])


def test_frame_section_filter(regression_report):
    """`frame(section=...)` returns only rows of the requested bucket."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize()

    issues_frame = result.frame(section="issue")
    assert set(issues_frame["code"]) >= {"TST001"}
    assert all(issues_frame["section"] == "issue")

    tips_frame = result.frame(section="tip")
    assert set(tips_frame["code"]) >= {"TST002"}
    assert all(tips_frame["section"] == "tip")

    passed_codes = set(result.frame(section="passed")["code"])
    assert "TST001" not in passed_codes
    assert "TST002" not in passed_codes

    assert set(result.frame()["code"]) >= {"TST001", "TST002"}


def test_header_reports_all_counts(regression_report):
    """The header reports issue, tip, passed, not applicable and ignored counts."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    result = regression_report.checks.summarize(ignore=["SKD001"])
    assert "issue(s)" in result._header
    assert "tip(s)" in result._header
    assert "passed" in result._header
    assert "not applicable" in result._header
    assert "1 ignored" in result._header


def test_html_tabs(regression_report):
    """The HTML repr contains one label per bucket with its count."""
    regression_report.checks.add([MockCheck(has_issue=True), TipCheck()])
    html = regression_report.checks.summarize()._repr_html_()
    assert "Issues (" in html
    assert "Tips (" in html
    assert "Passed (" in html
    assert "Not Applicable (" in html


class NotApplicableMockCheck(Check):
    code = "TSTNA"
    title = "Not applicable check"
    report_type = "estimator"
    docs_url = "tstna"

    def check_function(self, report):
        raise CheckNotApplicable("Mock check is not applicable.")


def test_not_applicable_goes_to_not_applicable_section(regression_report):
    """A check raising CheckNotApplicable appears under not applicable."""
    regression_report.checks.add([NotApplicableMockCheck()])
    result = regression_report.checks.summarize()
    na = result.frame(section="not_applicable").set_index("code")
    assert "TSTNA" in na.index
    assert na.loc["TSTNA", "explanation"] == "Mock check is not applicable."
    assert "TSTNA" not in set(result.frame(section="passed")["code"])
    assert "TSTNA" not in set(result.frame(section="issue")["code"])
    assert "TSTNA" not in set(result.frame(section="tip")["code"])


class SlowMockCheck(Check):
    code = "TSTSLOW"
    title = "Slow mock check"
    report_type = "estimator"
    docs_url = "tstslow"
    slow = True

    def __init__(self, *, fail=False):
        self.calls = 0
        self._fail = fail

    def check_function(self, report):
        if self._fail:
            raise AssertionError("slow check should not have been called")
        self.calls += 1
        return "Slow finding."


def test_summarize_fast_mode_skips_uncached_slow_checks(regression_report):
    """fast_mode=True skips slow checks that are not cached."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    codes = set(regression_report.checks.summarize(fast_mode=True).frame()["code"])
    assert "TSTSLOW" not in codes
    assert slow_check.calls == 0


def test_summarize_fast_mode_uses_cached_slow_results(regression_report):
    """fast_mode=True surfaces slow results that were already cached."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    regression_report.checks.summarize()
    assert slow_check.calls == 1
    issues = (
        regression_report.checks.summarize(fast_mode=True)
        .frame(section="issue")
        .set_index("code")
    )
    assert "TSTSLOW" in issues.index
    assert slow_check.calls == 1


def test_html_repr_does_not_compute_slow(regression_report):
    """The HTML repr never invokes slow check functions."""
    regression_report.checks.add([SlowMockCheck(fail=True)])
    fragments = regression_report._html_repr_fragments()
    assert "checks_summary" in fragments


def test_html_repr_shows_cached_slow(regression_report):
    """A cached slow result is reflected in the HTML repr summary."""
    slow_check = SlowMockCheck()
    regression_report.checks.add([slow_check])
    regression_report.checks.summarize()
    fragments = regression_report._html_repr_fragments()
    assert "1 issue(s)" in fragments["checks_summary"]


def test_subclass_check_without_slow_attr_treated_as_fast(regression_report):
    """Subclass of Check without `slow` inherits the protocol default."""

    class CheckNoSlowAttr(Check):
        code = "TSTFAST"
        title = "No slow attr"
        report_type = "estimator"
        docs_url = "tstfast"
        severity = "issue"

        def check_function(self, report):
            return "Found."

    check = CheckNoSlowAttr()
    assert check.slow is False
    regression_report.checks.add([check])
    codes = set(regression_report.checks.summarize(fast_mode=True).frame()["code"])
    assert "TSTFAST" in codes
