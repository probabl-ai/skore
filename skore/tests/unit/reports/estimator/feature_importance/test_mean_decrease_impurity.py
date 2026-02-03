import pandas as pd
import pytest
import sklearn
from sklearn.base import is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import EstimatorReport


@pytest.mark.parametrize(
    "data, estimator, expected_shape",
    [
        (
            make_classification(n_features=5, random_state=42),
            RandomForestClassifier(n_estimators=2, random_state=0),
            (5, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            RandomForestClassifier(n_estimators=2, random_state=0),
            (5, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            make_pipeline(
                StandardScaler(), RandomForestClassifier(n_estimators=2, random_state=0)
            ),
            (5, 1),
        ),
        (
            make_classification(n_features=5, random_state=42),
            make_pipeline(
                StandardScaler(), RandomForestClassifier(n_estimators=2, random_state=0)
            ),
            (5, 1),
        ),
        (
            make_regression(n_features=5, n_targets=3, random_state=42),
            RandomForestRegressor(n_estimators=2, random_state=0),
            (5, 1),
        ),
    ],
)
def test_numpy_arrays(data, estimator, expected_shape):
    X, y = data
    estimator.fit(X, y)
    report = EstimatorReport(estimator)
    result = report.inspection.impurity_decrease()

    assert result.shape == expected_shape

    expected_index = (
        [f"x{i}" for i in range(X.shape[1])]
        if isinstance(estimator, Pipeline)
        else [f"Feature #{i}" for i in range(X.shape[1])]
    )
    assert result.index.tolist() == expected_index

    expected_columns = ["Mean decrease impurity"]
    assert result.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(n_estimators=2, random_state=0),
        make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=2, random_state=0)
        ),
    ],
)
def test_pandas_dataframe(estimator):
    """If provided, the `impurity_decrease` dataframe uses the feature names."""
    X, y = make_classification(n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.inspection.impurity_decrease()

    assert result.shape == (5, 1)
    assert result.index.tolist() == [
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_4",
    ]
    assert result.columns.tolist() == ["Mean decrease impurity"]


def _make_estimator_param(estimator):
    return pytest.param(estimator, id=estimator.__class__.__name__)


@pytest.mark.parametrize(
    "estimator",
    [
        _make_estimator_param(estimator)
        for estimator in [
            sklearn.ensemble.AdaBoostClassifier(n_estimators=2),
            sklearn.ensemble.AdaBoostRegressor(n_estimators=2),
            sklearn.ensemble.ExtraTreesClassifier(n_estimators=2),
            sklearn.ensemble.ExtraTreesRegressor(n_estimators=2),
            sklearn.ensemble.GradientBoostingClassifier(n_estimators=2),
            sklearn.ensemble.GradientBoostingRegressor(n_estimators=2),
            sklearn.ensemble.RandomForestClassifier(n_estimators=2),
            sklearn.ensemble.RandomForestRegressor(n_estimators=2),
            sklearn.ensemble.RandomTreesEmbedding(n_estimators=2),
            sklearn.tree.DecisionTreeClassifier(),
            sklearn.tree.DecisionTreeRegressor(),
            sklearn.tree.ExtraTreeClassifier(),
            sklearn.tree.ExtraTreeRegressor(),
        ]
    ],
)
def test_all_sklearn_estimators(
    request, estimator, regression_data, binary_classification_data
):
    """Check that `impurity_decrease` is supported for every sklearn estimator."""
    if is_regressor(estimator):
        X, y = regression_data
    else:
        X, y = binary_classification_data

    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.inspection.impurity_decrease()

    assert result.shape == (X.shape[1], 1)
    assert result.index.tolist() == [f"Feature #{i}" for i in range(X.shape[1])]
    assert result.columns.tolist() == ["Mean decrease impurity"]


def test_pipeline_with_transformer(regression_data):
    """If the estimator is a pipeline containing a transformer that changes the
    features, adapt the feature names in the output table."""
    from sklearn.preprocessing import PolynomialFeatures

    X, y = regression_data
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(4)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=True),
        RandomForestRegressor(n_estimators=2, random_state=0),
    )

    report = EstimatorReport(
        model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    result = report.inspection.impurity_decrease()
    assert result.shape == (report.estimator_[-1].n_features_in_, 1)
    assert result.index.tolist() == [
        "1",
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_0 my_feature_1",
        "my_feature_0 my_feature_2",
        "my_feature_0 my_feature_3",
        "my_feature_1 my_feature_2",
        "my_feature_1 my_feature_3",
        "my_feature_2 my_feature_3",
    ]
    assert result.columns.tolist() == ["Mean decrease impurity"]
