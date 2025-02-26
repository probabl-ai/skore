import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport


def regression_data():
    X, y = make_regression(n_features=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
    return data


def regression_data_dataframe():
    data = regression_data()
    data["X_train"] = pd.DataFrame(
        data["X_train"], columns=["my_feature_0", "my_feature_1", "my_feature_2"]
    )
    data["X_test"] = pd.DataFrame(
        data["X_test"], columns=["my_feature_0", "my_feature_1", "my_feature_2"]
    )
    return data


def case_default_args_numpy():
    data = regression_data()

    kwargs = {"random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"Feature #{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_r2_numpy():
    data = regression_data()

    kwargs = {"scoring": make_scorer(r2_score), "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"Feature #{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_train_numpy():
    data = regression_data()

    kwargs = {"data_source": "train", "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"Feature #{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_several_scoring_numpy():
    data = regression_data()

    kwargs = {"scoring": ["r2", "neg_root_mean_squared_error"], "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((2 * 3, 5)),
        index=pd.MultiIndex.from_product(
            [
                ["r2", "neg_root_mean_squared_error"],
                (f"Feature #{i}" for i in range(3)),
            ],
            names=("Metric", "Feature"),
        ),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_default_args_dataframe():
    data = regression_data_dataframe()

    kwargs = {"random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"my_feature_{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_r2_dataframe():
    data = regression_data_dataframe()

    kwargs = {"scoring": make_scorer(r2_score), "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"my_feature_{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_train_dataframe():
    data = regression_data_dataframe()

    kwargs = {"data_source": "train", "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"my_feature_{i}" for i in range(3)), name="Feature"),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


def case_several_scoring_dataframe():
    data = regression_data_dataframe()

    kwargs = {"scoring": ["r2", "neg_root_mean_squared_error"], "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((2 * 3, 5)),
        index=pd.MultiIndex.from_product(
            [
                ["r2", "neg_root_mean_squared_error"],
                (f"my_feature_{i}" for i in range(3)),
            ],
            names=("Metric", "Feature"),
        ),
        columns=pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat"),
    )
    return data, kwargs, expected


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(LinearRegression(), id="linear_regression"),
        pytest.param(
            make_pipeline(StandardScaler(), LinearRegression()), id="pipeline"
        ),
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        case_default_args_numpy,
        case_r2_numpy,
        case_train_numpy,
        case_several_scoring_numpy,
        case_default_args_dataframe,
        case_r2_dataframe,
        case_train_dataframe,
        case_several_scoring_dataframe,
    ],
)
def test_estimator_report_feature_permutation(estimator, params):
    data, kwargs, expected = params()

    report = EstimatorReport(estimator, **data)
    result = report.feature_importance.feature_permutation(**kwargs)

    # Do not check values
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=np.inf)


def test_estimator_report_feature_permutation_cache(regression_data):
    """Results are properly cached."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    result = report.feature_importance.feature_permutation(data_source="train")
    assert report._cache != {}
    cached_result = report.feature_importance.feature_permutation(data_source="train")

    pd.testing.assert_frame_equal(cached_result, result)


def test_estimator_report_feature_permutation_not_fitted(regression_data):
    """If estimator is not fitted, raise"""
    X, y = regression_data

    report = EstimatorReport(
        LinearRegression(),
        fit=False,
        X_test=X,
        y_test=y,
    )

    error_msg = "This LinearRegression instance is not fitted yet"
    with pytest.raises(NotFittedError, match=error_msg):
        report.feature_importance.feature_permutation()
