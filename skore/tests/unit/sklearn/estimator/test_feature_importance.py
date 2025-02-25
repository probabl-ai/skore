import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport


@pytest.fixture
def regression_data_5_features():
    return make_regression(n_features=5, random_state=42)


def test_estimator_report_feature_importance_help(capsys, regression_data_5_features):
    """Check that the help method writes to the console."""
    X, y = regression_data_5_features
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    report.feature_importance.help()
    captured = capsys.readouterr()
    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out


def test_estimator_report_feature_importance_repr(regression_data_5_features):
    """Check that __repr__ returns a string starting with the expected prefix."""
    X, y = regression_data_5_features
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    repr_str = repr(report.feature_importance)
    assert "skore.EstimatorReport.feature_importance" in repr_str
    assert "report.feature_importance.help()" in repr_str


########################################################################################
# Check the coefficients feature importance metric
########################################################################################


@pytest.mark.parametrize(
    "data, estimator, column_base_name, expected_shape",
    [
        (
            make_regression(n_features=5, random_state=42),
            LinearRegression(),
            None,
            (6, 1),
        ),
        (
            make_classification(n_features=5, random_state=42),
            LogisticRegression(),
            None,
            (6, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            LogisticRegression(),
            "Class",
            (6, 3),
        ),
        (
            make_regression(n_features=5, random_state=42),
            Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
            None,
            (6, 1),
        ),
        (
            make_regression(n_features=5, n_targets=3, random_state=42),
            LinearRegression(),
            "Target",
            (6, 3),
        ),
    ],
)
def test_estimator_report_coefficients_numpy_arrays(
    data, estimator, column_base_name, expected_shape
):
    X, y = data
    estimator.fit(X, y)
    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()
    assert result.shape == expected_shape

    expected_index = ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    assert result.index.tolist() == expected_index

    expected_columns = (
        ["Coefficient"]
        if expected_shape[1] == 1
        else [f"{column_base_name} #{i}" for i in range(expected_shape[1])]
    )
    assert result.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        make_pipeline(StandardScaler(), LinearRegression()),
    ],
)
def test_estimator_report_coefficients_pandas_dataframe(estimator):
    """If provided, the coefficients dataframe uses the feature names."""
    X, y = make_regression(n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    assert result.shape == (6, 1)
    assert result.index.tolist() == [
        "Intercept",
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_4",
    ]
    assert result.columns.tolist() == ["Coefficient"]
