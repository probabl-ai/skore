import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, CrossValidationReport


@pytest.mark.parametrize(
    "estimator",
    [
        Ridge(),
        TransformedTargetRegressor(Ridge()),
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("transformed_ridge", TransformedTargetRegressor(Ridge())),
            ]
        ),
    ],
)
def test_with_model_exposing_coef(regression_data, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X, y = regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_data):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X, y = regression_data
    report = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    assert not hasattr(report.feature_importance, "coefficients")
