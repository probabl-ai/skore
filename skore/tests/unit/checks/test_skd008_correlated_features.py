import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from skrub import tabular_pipeline

from skore import evaluate
from skore.checks._utils import CheckNotApplicable
from skore.checks.model_checks import CheckCorrelatedFeatures


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "estimator", [LinearRegression(), tabular_pipeline(LinearRegression())]
)
def test_correlated_features(report_type, estimator):
    """SKD008 issue is emitted when two features are near-perfectly correlated."""
    rng = np.random.RandomState(42)
    X = rng.standard_normal((20, 4))
    X[:, 1] = X[:, 0] + rng.standard_normal(20) * 1e-4
    y = rng.standard_normal(20)
    report = evaluate(
        estimator,
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    explanation = CheckCorrelatedFeatures().check_function(report)
    assert explanation is not None
    assert "1 pair(s) of features" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_emitted_for_independent_features(report_type, regression_data):
    """SKD008 issue is absent when features are independent."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckCorrelatedFeatures().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_correlated_features_multioutput(report_type, regression_multioutput_data):
    """SKD008 is emitted for multioutput regression when features are correlated."""
    X, y = regression_multioutput_data
    rng = np.random.RandomState(42)
    X[:, 1] = X[:, 0] + rng.standard_normal(X.shape[0]) * 1e-4
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    assert CheckCorrelatedFeatures().check_function(report) is not None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_with_a_single_feature(report_type, regression_data):
    """SKD008 needs at least 2 features to compute pairwise correlations."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(),
        X[:, :1],
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    with pytest.raises(CheckNotApplicable, match="got 1"):
        CheckCorrelatedFeatures().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_correlation_with_exactly_two_numeric_features(
    report_type, regression_data
):
    """With exactly 2 numeric features, spearman returns a scalar; still checked."""
    rng = np.random.RandomState(42)
    X, y = regression_data
    X = X[:, :2].copy()
    X[:, 1] = X[:, 0] + rng.standard_normal(X.shape[0]) * 1e-4
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckCorrelatedFeatures().check_function(report)
    assert explanation is not None
    assert "1 pair(s) of features" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_with_a_single_numeric_feature(report_type, regression_data):
    """Non-numeric columns are excluded, so a lone numeric feature is not applicable."""
    X, y = regression_data
    X = pd.DataFrame(X[:, :1], columns=["num"])
    X["cat"] = ["a", "b"] * (len(X) // 2)
    pipeline = Pipeline(
        [
            ("select", ColumnTransformer([("num", "passthrough", ["num"])])),
            ("model", LinearRegression()),
        ]
    )
    report = evaluate(pipeline, X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable, match="got 1"):
        CheckCorrelatedFeatures().check_function(report)
