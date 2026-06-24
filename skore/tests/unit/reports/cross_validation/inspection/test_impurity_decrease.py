import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import CrossValidationReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize(
    "data_fixture,estimator",
    [
        (
            "binary_classification_data",
            RandomForestClassifier(random_state=0, n_estimators=5),
        ),
        (
            "binary_classification_data",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(random_state=0, n_estimators=5)),
                ]
            ),
        ),
        ("regression_data", RandomForestRegressor(random_state=0, n_estimators=5)),
        (
            "regression_data",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestRegressor(random_state=0, n_estimators=5)),
                ]
            ),
        ),
    ],
)
def test_with_model_exposing_feature_importances(data_fixture, estimator, request):
    """Check that we can create an impurity decrease display from model exposing a
    `feature_importances_` attribute."""
    X, y = request.getfixturevalue(data_fixture)
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.inspection, "impurity_decrease")
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)


def test_with_model_not_exposing_feature_importances(binary_classification_data):
    """Check that we cannot create an impurity decrease display from model not exposing
    a `feature_importances_` attribute."""
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=2)
    assert not hasattr(report.inspection, "impurity_decrease")
