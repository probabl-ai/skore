import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
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


repeat_columns = pd.Index((f"Repeat #{i}" for i in range(5)), name="Repeat")

multiindex_numpy = pd.MultiIndex.from_product(
    [["r2"], (f"Feature #{i}" for i in range(3))],
    names=("Metric", "Feature"),
)


def case_default_args_numpy():
    data = regression_data()

    kwargs = {"random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=multiindex_numpy,
        columns=repeat_columns,
    )

    return data, kwargs, expected


def case_r2_numpy():
    data = regression_data()

    kwargs = {"scoring": make_scorer(r2_score), "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"Feature #{i}" for i in range(3)), name="Feature"),
        columns=repeat_columns,
    )

    return data, kwargs, expected


def case_train_numpy():
    data = regression_data()

    kwargs = {"data_source": "train", "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=multiindex_numpy,
        columns=repeat_columns,
    )
    return data, kwargs, expected


def case_several_scoring_numpy():
    data = regression_data()

    kwargs = {"scoring": ["r2", "rmse"], "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((2 * 3, 5)),
        index=pd.MultiIndex.from_product(
            [
                ["r2", "rmse"],
                (f"Feature #{i}" for i in range(3)),
            ],
            names=("Metric", "Feature"),
        ),
        columns=repeat_columns,
    )
    return data, kwargs, expected


def case_X_y():
    data = regression_data()
    X, y = data["X_train"], data["y_train"]

    kwargs = {"data_source": "X_y", "X": X, "y": y, "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=multiindex_numpy,
        columns=repeat_columns,
    )
    return data, kwargs, expected


def case_aggregate():
    data = regression_data()

    kwargs = {"data_source": "train", "aggregate": "mean", "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 1)),
        index=multiindex_numpy,
        columns=pd.Index(["mean"]),
    )
    return data, kwargs, expected


def case_default_args_pandas():
    data = regression_data_dataframe()

    kwargs = {"random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.MultiIndex.from_product(
            [["r2"], (f"my_feature_{i}" for i in range(3))],
            names=("Metric", "Feature"),
        ),
        columns=repeat_columns,
    )
    return data, kwargs, expected


def case_r2_pandas():
    data = regression_data_dataframe()

    kwargs = {"scoring": make_scorer(r2_score), "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.Index((f"my_feature_{i}" for i in range(3)), name="Feature"),
        columns=repeat_columns,
    )

    return data, kwargs, expected


def case_train_pandas():
    data = regression_data_dataframe()

    kwargs = {"data_source": "train", "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((3, 5)),
        index=pd.MultiIndex.from_product(
            [["r2"], (f"my_feature_{i}" for i in range(3))],
            names=("Metric", "Feature"),
        ),
        columns=repeat_columns,
    )
    return data, kwargs, expected


def case_several_scoring_pandas():
    data = regression_data_dataframe()

    kwargs = {"scoring": ["r2", "rmse"], "random_state": 42}

    expected = pd.DataFrame(
        data=np.zeros((2 * 3, 5)),
        index=pd.MultiIndex.from_product(
            [
                ["r2", "rmse"],
                (f"my_feature_{i}" for i in range(3)),
            ],
            names=("Metric", "Feature"),
        ),
        columns=repeat_columns,
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
        case_aggregate,
        case_default_args_pandas,
        case_r2_pandas,
        case_train_pandas,
        case_several_scoring_pandas,
        case_X_y,
    ],
)
def test(estimator, params):
    data, kwargs, expected = params()

    report = EstimatorReport(estimator, **data)
    result = report.feature_importance.feature_permutation(**kwargs)

    # Do not check values
    pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=np.inf)


def test_cache_n_jobs(regression_data):
    """Results are properly cached. `n_jobs` is not in the cache."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    result = report.feature_importance.feature_permutation(
        data_source="train", random_state=42, n_jobs=1
    )
    assert report._cache != {}
    cached_result = report.feature_importance.feature_permutation(
        data_source="train", random_state=42, n_jobs=-1
    )
    assert len(report._cache) == 1

    pd.testing.assert_frame_equal(cached_result, result)


def test_cache_random_state(regression_data):
    """If `random_state` is not an int (the default is None)
    then the result is not cached.

    `random_state` must be an int or None.
    """

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    # random_state is None
    report.feature_importance.feature_permutation(data_source="train")
    # so no cache
    assert report._cache == {}

    # random_state is a RandomState
    err_msg = (
        "random_state must be an integer or None; "
        "got <class 'numpy.random.mtrand.RandomState'>"
    )
    with pytest.raises(ValueError, match=err_msg):
        report.feature_importance.feature_permutation(
            data_source="train",
            random_state=np.random.RandomState(42),
        )
    # so no cache
    assert report._cache == {}

    # random_state is an int
    result = report.feature_importance.feature_permutation(
        data_source="train", random_state=42
    )
    # so the result is cached
    assert report._cache != {}
    cached_result = report.feature_importance.feature_permutation(
        data_source="train", random_state=42
    )
    assert len(report._cache) == 1

    pd.testing.assert_frame_equal(cached_result, result)


def test_cache_scoring(regression_data):
    """`scoring` is in the cache."""

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    report.feature_importance.feature_permutation(
        data_source="train", scoring="r2", random_state=42
    )
    report.feature_importance.feature_permutation(
        data_source="train", scoring="rmse", random_state=42
    )
    # Scorings are different, so cache keys should be different
    assert len(report._cache) == 2


@pytest.mark.parametrize(
    "scoring",
    [
        make_scorer(r2_score),
        ["r2", "rmse"],
        {"r2": make_scorer(r2_score), "rmse": make_scorer(root_mean_squared_error)},
    ],
)
def test_cache_scoring_is_callable(regression_data, scoring):
    """If `scoring` is a callable then the result is cached properly."""

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    result = report.feature_importance.feature_permutation(
        data_source="train", scoring=scoring, random_state=42
    )
    assert report._cache != {}
    cached_result = report.feature_importance.feature_permutation(
        data_source="train", scoring=copy.copy(scoring), random_state=42
    )
    assert len(report._cache) == 1

    pd.testing.assert_frame_equal(cached_result, result)


def test_not_fitted(regression_data):
    """If estimator is not fitted, raise"""
    X, y = regression_data

    report = EstimatorReport(LinearRegression(), fit=False, X_test=X, y_test=y)

    error_msg = "This LinearRegression instance is not fitted yet"
    with pytest.raises(NotFittedError, match=error_msg):
        report.feature_importance.feature_permutation()
