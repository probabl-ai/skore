import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skore import EstimatorReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(n_estimators=2, random_state=0),
        DecisionTreeClassifier(random_state=0),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=2, random_state=0)),
            ]
        ),
    ],
)
def test_with_model_exposing_mean_decrease_impurity_classification(
    binary_classification_train_test_split, estimator
):
    """Check that we can create an impurity decrease display from classification model
    exposing a `feature_importances_` attribute."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.inspection, "impurity_decrease")
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestRegressor(n_estimators=2, random_state=0),
        DecisionTreeRegressor(random_state=0),
    ],
)
def test_with_model_exposing_mean_decrease_impurity_regression(
    regression_train_test_split, estimator
):
    """Check that we can create an impurity decrease display from regression model
    exposing a `feature_importances_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.inspection, "impurity_decrease")
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)


def test_with_model_not_exposing_mean_decrease_impurity(
    binary_classification_train_test_split,
):
    """Check that we cannot create an impurity decrease display from model not exposing a
    `feature_importances_` attribute."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        LogisticRegression(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not hasattr(report.inspection, "impurity_decrease")
