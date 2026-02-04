import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skore import EstimatorReport, ImpurityDecreaseDisplay


@pytest.fixture
def train_test_split(request):
    """Return the train/test split identified by the parametrized fixture name."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "train_test_split,estimator",
    [
        (
            "binary_classification_train_test_split",
            DecisionTreeClassifier(max_depth=2),
        ),
        (
            "binary_classification_train_test_split",
            make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2)),
        ),
        (
            "regression_train_test_split",
            DecisionTreeRegressor(max_depth=2),
        ),
        (
            "regression_train_test_split",
            make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=2)),
        ),
    ],
    indirect=["train_test_split"],
)
def test_with_model_exposing_mean_decrease_impurity(train_test_split, estimator):
    """Check that we can create an impurity decrease display from a decision tree model
    (alone or in a pipeline) exposing a `feature_importances_` attribute."""
    X_train, X_test, y_train, y_test = train_test_split
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
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not hasattr(report.inspection, "impurity_decrease")
