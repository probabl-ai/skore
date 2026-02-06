import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import ComparisonReport, CrossValidationReport, ImpurityDecreaseDisplay


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
def test_with_model_exposing_feature_importances(regression_data, estimator):
    """Check that we can create an impurity decrease display from model exposing a
    `feature_importances_` attribute."""
    X, y = regression_data
    report_1 = CrossValidationReport(estimator, X, y, splitter=2)
    report_2 = CrossValidationReport(estimator, X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "impurity_decrease")
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)


def test_with_model_not_exposing_feature_importances(regression_data):
    """
    Check that we cannot create an impurity decrease display from model not exposing
    a `feature_importances_` attribute.
    """
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.inspection, "impurity_decrease")


def test_with_mixed_reports(regression_data):
    """Check that we cannot create an impurity decrease display from mixed reports."""
    X, y = regression_data
    report_1 = CrossValidationReport(
        RandomForestRegressor(random_state=0), X, y, splitter=2
    )
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.inspection, "impurity_decrease")
