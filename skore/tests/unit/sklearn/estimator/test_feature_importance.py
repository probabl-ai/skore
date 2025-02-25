import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport


def assert_frame_shape_equal(result, expected):
    """Assert that `result` and `expected` have the same shape and names.

    Do not check the data.
    """
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=np.inf)


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
    "data, estimator, expected",
    [
        (
            "regression_data_5_features",
            LinearRegression(),
            pd.DataFrame(
                data=[0.0] * 6,
                index=[
                    "Intercept",
                    "Feature #0",
                    "Feature #1",
                    "Feature #2",
                    "Feature #3",
                    "Feature #4",
                ],
                columns=["Coefficient"],
            ),
        ),
        (
            make_classification(n_features=5, random_state=42),
            LogisticRegression(),
            pd.DataFrame(
                data=[0.0] * 6,
                index=[
                    "Intercept",
                    "Feature #0",
                    "Feature #1",
                    "Feature #2",
                    "Feature #3",
                    "Feature #4",
                ],
                columns=["Coefficient"],
            ),
        ),
        (
            make_classification(
                n_features=5, n_classes=3, n_samples=30, n_informative=3
            ),
            LogisticRegression(),
            pd.DataFrame(
                data=[[0.0] * 3] * 6,
                index=[
                    "Intercept",
                    "Feature #0",
                    "Feature #1",
                    "Feature #2",
                    "Feature #3",
                    "Feature #4",
                ],
                columns=["Target #0", "Target #1", "Target #2"],
            ),
        ),
        (
            "regression_data_5_features",
            Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
            pd.DataFrame(
                data=[0.0] * 6,
                index=[
                    "Intercept",
                    "Feature #0",
                    "Feature #1",
                    "Feature #2",
                    "Feature #3",
                    "Feature #4",
                ],
                columns=["Coefficient"],
            ),
        ),
        (
            make_regression(n_features=5, n_targets=3, random_state=42),
            LinearRegression(),
            pd.DataFrame(
                data=[[0.0] * 3] * 6,
                index=[
                    "Intercept",
                    "Feature #0",
                    "Feature #1",
                    "Feature #2",
                    "Feature #3",
                    "Feature #4",
                ],
                columns=["Target #0", "Target #1", "Target #2"],
            ),
        ),
    ],
)
def test_estimator_report_coefficients_numpy_arrays(request, data, estimator, expected):
    if isinstance(data, str):
        data = request.getfixturevalue(data)
    X, y = data

    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    assert_frame_shape_equal(result, expected)


def test_estimator_report_coefficients_pandas_dataframe(regression_data_5_features):
    """If provided, the model weights dataframe uses the feature names, where the
    estimator is a single estimator (not a pipeline)."""
    X, y = regression_data_5_features
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator = LinearRegression().fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    expected = pd.DataFrame(
        data=[0.0] * 6,
        index=[
            "Intercept",
            "my_feature_0",
            "my_feature_1",
            "my_feature_2",
            "my_feature_3",
            "my_feature_4",
        ],
        columns=["Coefficient"],
    )
    assert_frame_shape_equal(result, expected)


def test_estimator_report_coefficients_pandas_dataframe_pipeline(
    regression_data_5_features,
):
    """If provided, the model weights dataframe uses the feature names, where the
    estimator is a pipeline (not a single estimator)."""
    X, y = regression_data_5_features
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    expected = pd.DataFrame(
        data=[0.0] * 6,
        index=[
            "Intercept",
            "my_feature_0",
            "my_feature_1",
            "my_feature_2",
            "my_feature_3",
            "my_feature_4",
        ],
        columns=["Coefficient"],
    )
    assert_frame_shape_equal(result, expected)
