import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, EstimatorReport


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
def test_with_model_exposing_coef(regression_train_test_split, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.inspection, "coefficients")
    display = report.inspection.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_train_test_split):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not hasattr(report.inspection, "coefficients")
