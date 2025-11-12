import contextlib
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from skore import EstimatorReport, train_test_split
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def regression_data() -> dict:
    X, y = make_regression(n_features=3, random_state=42)
    split_data = train_test_split(X, y, test_size=0.2, random_state=42, as_dict=True)
    return split_data


def regression_data_dataframe():
    data = regression_data()
    columns = ["my_feature_0", "my_feature_1", "my_feature_2"]
    data["X_train"] = pd.DataFrame(data["X_train"], columns=columns)
    data["X_test"] = pd.DataFrame(data["X_test"], columns=columns)
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


class TestAtStep:
    @pytest.fixture(params=["numpy", "dataframe"])
    def split_data(self, request):
        array_type = request.param

        if array_type == "numpy":
            return regression_data()
        elif array_type == "dataframe":
            return regression_data_dataframe()

    @pytest.fixture
    def pipeline_report(self, split_data) -> EstimatorReport:
        pipeline = make_pipeline(
            StandardScaler(), PCA(n_components=2), LinearRegression()
        )
        return EstimatorReport(pipeline, **split_data)

    @pytest.mark.parametrize("at_step", [0, -1, 1])
    def test_int(self, pipeline_report, at_step):
        """
        Test the `at_step` integer parameter for permutation importance with a pipeline.
        """
        result = pipeline_report.feature_importance.permutation(
            seed=42, at_step=at_step
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.nlevels == 2
        assert result.index.names == ["Metric", "Feature"]
        assert result.shape[0] > 0

    def test_str(self, pipeline_report):
        """
        Test the `at_step` string parameter for permutation importance with a pipeline.
        """
        result = pipeline_report.feature_importance.permutation(seed=42, at_step="pca")

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.nlevels == 2
        assert result.index.names == ["Metric", "Feature"]
        assert result.shape[0] > 0

    def test_non_pipeline(self, split_data):
        """
        For non-pipeline estimators, changing at_step should not change the results.
        """
        report = EstimatorReport(LinearRegression(), **split_data)

        result_start = report.feature_importance.permutation(seed=42, at_step=0)
        result_end = report.feature_importance.permutation(seed=42, at_step=-1)

        pd.testing.assert_frame_equal(result_start, result_end)

    AT_STEP_TOO_LARGE = (
        "at_step must be strictly smaller in magnitude than "
        "the number of steps in the Pipeline"
    )

    @pytest.mark.parametrize(
        "at_step, err_msg",
        [
            (8, AT_STEP_TOO_LARGE),
            (-8, AT_STEP_TOO_LARGE),
            ("hello", "'hello' is not in list"),
            (0.5, "at_step must be an integer or a string"),
        ],
    )
    def test_value_error(self, pipeline_report, at_step, err_msg):
        """If `at_step` value is not appropriate, a ValueError is raised."""
        with pytest.raises(ValueError, match=err_msg):
            pipeline_report.feature_importance.permutation(seed=42, at_step=at_step)

    def test_sparse_array(self, split_data):
        """If one of the steps outputs a sparse array, `permutation` still works."""
        pipeline = make_pipeline(
            SplineTransformer(sparse_output=True), LinearRegression()
        )
        report = EstimatorReport(pipeline, **split_data)

        report.feature_importance.permutation(seed=42, at_step=-1)

    def test_feature_names(self, split_data):
        """If the requested pipeline step gives proper feature names,
        these names should appear in the output."""
        pipeline = make_pipeline(SplineTransformer(), LinearRegression())
        report = EstimatorReport(pipeline, **split_data)

        result = report.feature_importance.permutation(seed=42, at_step=-1)
        last_step_feature_names = list(report.estimator_[0].get_feature_names_out())
        assert list(result.index.levels[1]) == last_step_feature_names

    @pytest.mark.parametrize("at_step", [0, -1])
    def test_non_sklearn_pipeline(self, split_data, at_step):
        """If the pipeline contains non-sklearn-compliant transformers,
        `permutation` still works."""

        class Scaler(TransformerMixin, BaseEstimator):
            def fit(self, X, y=None):
                self.mean_ = X.mean(axis=0)
                self.std_ = X.std(axis=0)
                return self

            def transform(self, X, y=None):
                X_ = X.copy()
                return (X_ - self.mean_) / self.std_

        class Regressor(RegressorMixin, BaseEstimator):
            _is_fitted = False

            def fit(self, X, y):
                self.y_mean = np.mean(y)
                self._is_fitted = True
                return self

            def predict(self, X):
                return np.full(X.shape[0], self.y_mean)

            def __sklearn_is_fitted__(self):
                return self._is_fitted

        pipeline = make_pipeline(Scaler(), Regressor())
        report = EstimatorReport(pipeline, **split_data)

        report.feature_importance.permutation(seed=42, at_step=at_step)
