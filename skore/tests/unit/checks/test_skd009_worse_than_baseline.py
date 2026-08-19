import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from skrub import tabular_pipeline

from skore import evaluate
from skore.checks.model_checks import CheckWorseThanBaseline


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_detects_worse_than_baseline(report_type, regression_data):
    """Check that the worse-than-baseline tip is raised on a dummy estimator."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckWorseThanBaseline().check_function(report)
    assert explanation is not None
    assert "significantly worse than a HistGradientBoosting baseline" in explanation
    assert "Baseline performance on the test set" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_does_not_flag_same_model_as_baseline(report_type, regression_data):
    """SKD009 does not flag the model as worse when it is the baseline itself."""
    X, y = regression_data
    report = evaluate(
        tabular_pipeline(HistGradientBoostingRegressor()),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    explanation = CheckWorseThanBaseline().check_function(report)
    assert explanation is not None
    assert "significantly worse" not in explanation
    assert "on par with or better than a HistGradientBoosting baseline" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_shows_baseline_for_reference_on_strong_model(report_type):
    """SKD009 reports the baseline for reference when the model beats it."""
    X, y = make_regression(n_features=4, noise=0.1, random_state=0)
    report = evaluate(
        RidgeCV(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckWorseThanBaseline().check_function(report)
    assert explanation is not None
    assert "on par with or better than a HistGradientBoosting baseline" in explanation
    assert "for reference" in explanation


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
