import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import ComparisonReport, EstimatorReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestRegressor(random_state=0),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(random_state=0)),
            ]
        ),
    ],
)
def test_with_model_exposing_feature_importances(
    regression_train_test_split, estimator
):
    """Check that we can create an impurity decrease display from model exposing a
    `feature_importances_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "impurity_decrease")
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)


def test_with_model_not_exposing_feature_importances(regression_train_test_split):
    """
    Check that we cannot create an impurity decrease display from model not exposing a
    `feature_importances_` attribute.
    """
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.inspection, "impurity_decrease")


def test_with_mixed_reports(regression_train_test_split):
    """Check that we cannot create an impurity decrease display from mixed reports."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        RandomForestRegressor(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.inspection, "impurity_decrease")
