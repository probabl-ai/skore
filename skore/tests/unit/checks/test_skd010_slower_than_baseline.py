import math

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor

from skore import evaluate
from skore.checks.model_checks import CheckSlowerThanBaseline


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

    monkeypatch.setattr("skore.checks.model_checks.get_fit_time", fake_get_fit_time)
    monkeypatch.setattr(
        "skore.checks.model_checks.get_predict_time", fake_get_predict_time
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
