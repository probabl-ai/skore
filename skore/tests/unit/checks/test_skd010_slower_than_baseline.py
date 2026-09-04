import math

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor

from skore import CrossValidationReport, EstimatorReport, evaluate
from skore._checks.skd010_slower_than_baseline import (
    CheckSlowerThanBaseline,
    get_fit_time,
)
from skore._checks.utils import CheckNotApplicable


@pytest.fixture
def small_estimator_report(regression_data):
    X, y = regression_data
    return EstimatorReport(
        LinearRegression(), X_train=X[:60], y_train=y[:60], X_test=X[60:], y_test=y[60:]
    )


@pytest.fixture
def small_cv_report(regression_data):
    X, y = regression_data
    return CrossValidationReport(LinearRegression(), X=X, y=y, splitter=3)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_slower_than_baseline(report_type, regression_data):
    """Check that SKD010 is detected when the model is slower with similar scores."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    if report_type == "estimator":
        report._fit_time = math.inf
    else:
        report.reports_[0]._fit_time = math.inf

    explanation = CheckSlowerThanBaseline().check_function(report)
    assert explanation is not None
    assert "slower than a fast linear baseline" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_not_detected_when_gap_below_floor(monkeypatch, report_type, regression_data):
    """SKD010 does not fire when the ratio is high but the absolute gap is tiny."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )

    def fake_get_fit_time(r):
        # 5x ratio while under the 1s floor.
        return 0.5 if r is report else 0.1

    def fake_get_predict_time(_):
        return 0.01

    monkeypatch.setattr(
        "skore._checks.skd010_slower_than_baseline.get_fit_time", fake_get_fit_time
    )
    monkeypatch.setattr(
        "skore._checks.skd010_slower_than_baseline.get_predict_time",
        fake_get_predict_time,
    )

    assert CheckSlowerThanBaseline().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_detected_for_fast_model(report_type, regression_data):
    """Check that SKD010 does not fire when the model is not slower than baseline."""
    X, y = regression_data
    report = evaluate(
        RidgeCV(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckSlowerThanBaseline().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_detected_when_slower_model_scores_better(report_type):
    """SKD010 does not fire when a slower model has significantly better scores."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 4))
    y = np.sin(X[:, 0] * 5) + np.cos(X[:, 1] * 3) + rng.normal(scale=0.01, size=300)
    report = evaluate(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    if report_type == "estimator":
        report._fit_time = math.inf
    else:
        report.reports_[0]._fit_time = math.inf
    assert CheckSlowerThanBaseline().check_function(report) is None


def test_get_fit_time_estimator_report(small_estimator_report):
    assert get_fit_time(small_estimator_report) == small_estimator_report._fit_time
    assert get_fit_time(small_estimator_report) > 0


def test_get_fit_time_cv_report_is_mean_across_splits(small_cv_report):
    expected = float(
        small_cv_report.metrics.timings(aggregate="mean").loc["Fit time (s)"]
    )
    assert get_fit_time(small_cv_report) == expected


def test_get_fit_time_raises_not_applicable_for_prefit_estimator(regression_data):
    """A prefit estimator skips `_fit_estimator`, so `_fit_time` stays None."""
    X, y = regression_data
    fitted = LinearRegression().fit(X, y)
    report = EstimatorReport(fitted, X_test=X, y_test=y)
    assert report._fit_time is None
    with pytest.raises(CheckNotApplicable, match="Fit time is unavailable"):
        get_fit_time(report)
