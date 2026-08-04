import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV

from skore import evaluate
from skore._sklearn._checks.model_checks import CheckWorseThanBaseline


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_worse_than_baseline(report_type, regression_data):
    """Check that the worse-than-baseline issue is detected on a dummy estimator."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckWorseThanBaseline().check_function(report)
    assert explanation is not None
    assert (
        "not significantly better than a HistGradientBoosting baseline" in explanation
    )


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_detected_on_strong_model(report_type):
    """Check that SKD009 is not detected when the model beats HistGradientBoosting."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(
        RidgeCV(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckWorseThanBaseline().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_worse_than_baseline_multioutput(
    report_type, regression_multioutput_data
):
    """SKD009 emitted for multioutput regression when model is worse than baseline."""
    X, y = regression_multioutput_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckWorseThanBaseline().check_function(report) is not None
