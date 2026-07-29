import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from skore import evaluate
from skore._sklearn._checks.model_checks import CheckMDIHighCardinalityBias


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestRegressor(n_estimators=5, random_state=0),
        Pipeline([("rf", RandomForestRegressor(n_estimators=5, random_state=0))]),
    ],
)
def test_mdi_bias_with_high_cardinality(report_type, regression_data, estimator):
    """SKD007 tip is emitted with continuous features and tree importances."""
    X, y = regression_data
    report = evaluate(
        estimator, X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckMDIHighCardinalityBias().check_function(report)
    assert explanation is not None
    assert (
        "High-cardinality features detected: Feature 0, Feature 1, Feature 2 "
        "(and 1 more)" in explanation
    )


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_emitted_for_binary_features(report_type):
    """SKD007 tip is absent when all features are low-cardinality."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 2, size=(20, 4)).astype(float)
    y = rng.standard_normal(20)
    report = evaluate(
        RandomForestRegressor(n_estimators=5, random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    assert CheckMDIHighCardinalityBias().check_function(report) is None
