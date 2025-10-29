import contextlib
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import EstimatorReport
from skore._utils._testing import check_cache_changed, check_cache_unchanged


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

multi_index_numpy = pd.MultiIndex.from_product(
    [["r2"], (f"Feature #{i}" for i in range(3))],
    names=("Metric", "Feature"),
)

multi_index_pandas = pd.MultiIndex.from_product(
    [["r2"], (f"my_feature_{i}" for i in range(3))],
    names=("Metric", "Feature"),
)


def case_default_args_numpy():
    data = regression_data()

    kwargs = {"seed": 42}

    return data, kwargs, multi_index_numpy, repeat_columns


def case_r2_numpy():
    data = regression_data()

    kwargs = {"scoring": make_scorer(r2_score), "seed": 42}

    return (
        data,
        kwargs,
        pd.Index((f"Feature #{i}" for i in range(3)), name="Feature"),
        repeat_columns,
    )


def case_train_numpy():
    data = regression_data()

    kwargs = {"data_source": "train", "scoring": "r2", "seed": 42}

    return data, kwargs, multi_index_numpy, repeat_columns


def case_several_scoring_numpy():
    data = regression_data()

    kwargs = {"scoring": ["r2", "rmse"], "seed": 42}

    expected_index = pd.MultiIndex.from_product(
        [
            ["r2", "rmse"],
            (f"Feature #{i}" for i in range(3)),
        ],
        names=("Metric", "Feature"),
    )

    return data, kwargs, expected_index, repeat_columns


def case_X_y():
    data = regression_data()
    X, y = data["X_train"], data["y_train"]

    kwargs = {"data_source": "X_y", "X": X, "y": y, "seed": 42}

    return data, kwargs, multi_index_numpy, repeat_columns


def case_aggregate():
    data = regression_data()

    kwargs = {"data_source": "train", "aggregate": "mean", "seed": 42}

    return data, kwargs, multi_index_numpy, pd.Index(["mean"], dtype="object")


def case_default_args_pandas():
    data = regression_data_dataframe()

    kwargs = {"seed": 42}

    return data, kwargs, multi_index_pandas, repeat_columns


def case_r2_pandas():
    data = regression_data_dataframe()

    kwargs = {"scoring": make_scorer(r2_score), "seed": 42}

    return (
        data,
        kwargs,
        pd.Index((f"my_feature_{i}" for i in range(3)), name="Feature"),
        repeat_columns,
    )


def case_train_pandas():
    data = regression_data_dataframe()

    kwargs = {"data_source": "train", "seed": 42}

    return data, kwargs, multi_index_pandas, repeat_columns


def case_several_scoring_pandas():
    data = regression_data_dataframe()

    kwargs = {"scoring": ["r2", "rmse"], "seed": 42}

    expected_index = pd.MultiIndex.from_product(
        [
            ["r2", "rmse"],
            (f"my_feature_{i}" for i in range(3)),
        ],
        names=("Metric", "Feature"),
    )

    return data, kwargs, expected_index, repeat_columns


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
    data, kwargs, expected_index, expected_columns = params()

    report = EstimatorReport(estimator, **data)
    result = report.feature_importance.permutation(**kwargs)

    # Do not check values
    pd.testing.assert_index_equal(result.index, expected_index)
    pd.testing.assert_index_equal(result.columns, expected_columns)


def test_cache_n_jobs(regression_data):
    """Results are properly cached. `n_jobs` is not in the cache."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    with check_cache_changed(report._cache):
        result = report.feature_importance.permutation(
            data_source="train", seed=42, n_jobs=1
        )

    with check_cache_unchanged(report._cache):
        cached_result = report.feature_importance.permutation(
            data_source="train", seed=42, n_jobs=-1
        )

    pd.testing.assert_frame_equal(cached_result, result)


def test_cache_seed_error(regression_data):
    """Check that we only accept int and None as value for `seed`."""

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)
    assert report._cache == {}

    err_msg = "seed must be an integer or None"
    with pytest.raises(ValueError, match=err_msg):
        report.feature_importance.permutation(
            data_source="train", seed=np.random.RandomState(42)
        )


def test_cache_seed_none(regression_data):
    """Check the strategy on how we use the cache when `seed` is None.

    In this case, we store the result in the cache for sending to the hub but we
    always retrigger the computation.
    """

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)
    assert report._cache == {}

    importance_first_call = report.feature_importance.permutation(data_source="train")
    assert report._cache != {}
    importance_second_call = report.feature_importance.permutation(data_source="train")
    # the dataframes should be different
    with contextlib.suppress(AssertionError):
        pd.testing.assert_frame_equal(importance_first_call, importance_second_call)
    # the cache should contain the last result
    assert len(report._cache) == 1
    key = next(iter(report._cache.keys()))
    pd.testing.assert_frame_equal(report._cache[key], importance_second_call)


def test_cache_seed_int(regression_data):
    """Check the strategy on how we use the cache when `seed` is an int.

    In this case, we store and reload from the cache
    """
    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)
    assert report._cache == {}

    importance_first_call = report.feature_importance.permutation(
        data_source="train", seed=42
    )
    assert report._cache != {}
    importance_second_call = report.feature_importance.permutation(
        data_source="train", seed=42
    )
    # the dataframes should be the same
    pd.testing.assert_frame_equal(importance_first_call, importance_second_call)
    # the cache should contain the last result
    assert len(report._cache) == 1
    key = next(iter(report._cache.keys()))
    pd.testing.assert_frame_equal(report._cache[key], importance_second_call)


def test_cache_scoring(regression_data):
    """`scoring` is in the cache."""

    X, y = regression_data
    report = EstimatorReport(LinearRegression(), X_train=X, y_train=y)

    report.feature_importance.permutation(data_source="train", scoring="r2", seed=42)

    # Scorings are different, so cache keys should be different
    with check_cache_changed(report._cache):
        report.feature_importance.permutation(
            data_source="train", scoring="rmse", seed=42
        )


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

    with check_cache_changed(report._cache):
        result = report.feature_importance.permutation(
            data_source="train", scoring=scoring, seed=42
        )

    with check_cache_unchanged(report._cache):
        cached_result = report.feature_importance.permutation(
            data_source="train", scoring=copy.copy(scoring), seed=42
        )

    pd.testing.assert_frame_equal(cached_result, result)


def test_classification(binary_classification_data):
    """If `scoring` is a callable then the result is cached properly."""

    X, y = binary_classification_data
    report = EstimatorReport(LogisticRegression(), X_train=X, y_train=y)

    result = report.feature_importance.permutation(data_source="train", seed=42)

    pd.testing.assert_index_equal(
        result.index,
        pd.MultiIndex.from_product(
            [["accuracy"], (f"Feature #{i}" for i in range(X.shape[1]))],
            names=("Metric", "Feature"),
        ),
    )
    pd.testing.assert_index_equal(result.columns, repeat_columns)


def test_not_fitted(regression_data):
    """If estimator is not fitted, raise"""
    X, y = regression_data

    report = EstimatorReport(LinearRegression(), fit=False, X_test=X, y_test=y)

    error_msg = "This LinearRegression instance is not fitted yet"
    with pytest.raises(NotFittedError, match=error_msg):
        report.feature_importance.permutation()
