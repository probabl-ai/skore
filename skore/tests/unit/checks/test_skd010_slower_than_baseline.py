import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from skore import EstimatorReport, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckSlowerThanBaseline


def test_detects_slower_than_baseline(regression_data):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    X, y = regression_data
    report = evaluate(RandomForestRegressor(n_estimators=200, random_state=0), X, y)
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD010" in issues.index
    assert "slower than a fast linear baseline" in issues.loc["SKD010", "explanation"]


@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_slower_than_baseline_cv(regression_data):
    """Check that SKD010 is detected on a cross-validation report."""
    X, y = regression_data
    report = evaluate(
        RandomForestRegressor(n_estimators=200, random_state=0), X, y, splitter=3
    )
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD010" in issues.index
    assert "slower than a fast linear baseline" in issues.loc["SKD010", "explanation"]


def test_not_detected_for_fast_model(regression_data):
    """Check that SKD010 does not fire when the model is not slower than baseline."""
    X, y = regression_data
    report = evaluate(RidgeCV(), X, y)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD010" not in codes


def test_not_detected_when_slower_model_scores_better():
    """SKD010 does not fire when a slower model has significantly better scores."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 4))
    y = np.sin(X[:, 0] * 5) + np.cos(X[:, 1] * 3) + rng.normal(scale=0.01, size=300)
    report = evaluate(RandomForestRegressor(n_estimators=300, random_state=0), X, y)
    summary = report.checks.summarize()
    assert "SKD010" not in set(summary.frame(section="issue")["code"])
    assert "SKD010" in set(summary.frame(section="passed")["code"])


def test_not_applicable_when_train_data_missing(regression_train_test_split):
    """SKD010 needs train data to build the fast-baseline comparison."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    estimator = LinearRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    with pytest.raises(CheckNotApplicable):
        CheckSlowerThanBaseline().check_function(report)


def test_not_applicable_when_baseline_report_creation_fails(
    regression_data, monkeypatch
):
    """SKD010 raises when the fast linear baseline report can't be fit."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckSlowerThanBaseline().check_function(report)


def test_not_applicable_when_baseline_report_creation_fails_cv(
    regression_data, monkeypatch
):
    """SKD010 raises on a CV report when the baseline report can't be fit."""
    X, y = regression_data
    cv_report = evaluate(LinearRegression(), X, y, splitter=3)

    def failing_fit(self, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(EstimatorReport, "_fit_estimator", failing_fit)
    with pytest.raises(CheckNotApplicable):
        CheckSlowerThanBaseline().check_function(cv_report)
