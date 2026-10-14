from itertools import product

import numpy as np
import pandas as pd
import pytest
import skrub
from mlflow.data.numpy_dataset import NumpyDataset
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.polars_dataset import PolarsDataset
from numpy.testing import assert_array_equal
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.mlflow.reports import (
    Artifact,
    Metric,
    Model,
    _dataset_from_Xy,
    _sample_input_example,
    iter_cv,
    iter_cv_metrics,
    iter_estimator,
    iter_estimator_metrics,
)

REPORT_FIXTURES = ["clf_report", "mclf_report", "reg_report", "mreg_report"]
CV_REPORT_FIXTURES = [
    "cv_clf_report",
    "cv_mclf_report",
    "cv_reg_report",
    "cv_mreg_report",
]


X_pandas = pd.DataFrame(
    {
        "cat": pd.Series(["a", "b", "c"], dtype="category"),
        "num": [1, 2, 3],
    }
)
X_numpy = np.identity(3)
y_numpy = np.arange(3)
y_pandas = pd.Series(["no cancer", "cancer", "no cancer"])
y_pandas_multi_targets = pd.DataFrame(
    {
        "label": pd.Series(["no cancer", "cancer", "no cancer"]),
        "confidence": [0.9, 0.6, 0.95],
    }
)


@pytest.fixture
def report(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_metrics_smoke(report):
    assert all(
        isinstance(obj, Artifact | Metric) for obj in iter_estimator_metrics(report)
    )


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_metrics_smoke(report):
    assert all(isinstance(obj, Artifact | Metric) for obj in iter_cv_metrics(report))


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_smoke(report):
    assert len({type(obj) for obj in iter_estimator(report)}) >= 3


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_smoke(report):
    assert len({type(obj) for obj in iter_cv(report)}) >= 5


def test_sample_input_example_casts_category_to_object() -> None:
    sample = _sample_input_example(X_pandas, max_samples=2)

    assert sample.shape == (2, 2)
    assert not isinstance(sample["cat"].dtype, pd.CategoricalDtype)
    assert sample["cat"].tolist() == ["a", "b"]


def test_dataset_from_Xy_expands_multidimensional_array_targets():
    """A dataframe paired with a 2D array target becomes one column per output."""
    y = np.column_stack([y_numpy, y_numpy])

    dataset = _dataset_from_Xy(X_pandas, y).dataset

    assert isinstance(dataset, NumpyDataset)
    assert set(dataset.targets) == {"target_0", "target_1"}


def test_dataset_from_Xy_uses_single_frame_column_name():
    """A one-column target frame keeps its column name as the MLflow target."""
    y = y_pandas.to_frame(name="label")

    dataset = _dataset_from_Xy(X_pandas, y).dataset

    assert isinstance(dataset, PandasDataset)
    assert "label" in dataset.df.columns


@pytest.mark.parametrize(
    ("X", "y"),
    list(product([X_pandas, X_numpy], [y_pandas, y_numpy, y_pandas_multi_targets])),
)
def test_dataset_from_Xy(X, y):
    dataset = _dataset_from_Xy(X, y).dataset
    assert isinstance(dataset, (PandasDataset, NumpyDataset))

    if isinstance(dataset, NumpyDataset):
        assert_array_equal(dataset.features.shape, X.shape)
        if isinstance(dataset.targets, dict):
            for key, value in dataset.targets.items():
                assert_array_equal(value, y[key])
        else:
            assert_array_equal(dataset.targets, y)

    if isinstance(dataset, PandasDataset):
        target_col = getattr(y, "name", None) or "target"
        assert_array_equal(dataset.df.drop(columns=[target_col]), X)
        assert_array_equal(dataset.df[target_col], y)


def _top_level_model(items) -> Model:
    return next(item for item in items if isinstance(item, Model))


def test_iter_cv_skrub_learner_fits_environment() -> None:
    """Cross-validation logging fits a SkrubLearner on its environment."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    learner = skrub.X(X).skb.apply(DummyRegressor(), y=skrub.y(y)).skb.make_learner()
    report = CrossValidationReport(
        learner,
        data={"_skrub_X": X, "_skrub_y": y},
        splitter=KFold(n_splits=2),
    )

    model = _top_level_model(iter_cv(report))

    assert model.input_example is None
    assert model.model is not report.estimator_
    assert model.model.__sklearn_is_fitted__()
    assert not report.estimator_.__sklearn_is_fitted__()


def test_iter_cv_skrub_data_op_fits_environment() -> None:
    """A DataOp is logged through its learner, not ``fit(X, y)`` on the op."""
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    data_op = skrub.X(X).skb.apply(DummyRegressor(), y=skrub.y(y))
    report = CrossValidationReport(data_op, splitter=2)

    assert isinstance(report.estimator, skrub.DataOp)

    model = _top_level_model(iter_cv(report))

    assert model.input_example is None
    assert model.model.__sklearn_is_fitted__()


def test_iter_estimator_skrub_learner_has_no_input_example() -> None:
    """Estimator logging omits the input example for a SkrubLearner."""
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    learner = skrub.X().skb.apply(DummyRegressor(), y=skrub.y()).skb.make_learner()
    report = EstimatorReport(
        learner,
        train_data={"X": X_train, "y": y_train},
        test_data={"X": X_test, "y": y_test},
    )

    model = _top_level_model(iter_estimator(report))

    assert model.input_example is None
    assert model.model.__sklearn_is_fitted__()


@pytest.mark.parametrize(
    "y",
    [
        "series",
        "frame",
    ],
)
def test_dataset_from_Xy_polars(y: str) -> None:
    """Polars features and targets are recorded as a polars MLflow dataset."""
    import polars as pl

    features = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    if y == "series":
        target = pl.Series("label", [0, 1, 0])
    else:
        target = pl.DataFrame({"label": [0, 1, 0]})

    dataset = _dataset_from_Xy(features, target).dataset

    assert isinstance(dataset, PolarsDataset)
    assert list(dataset.df.columns) == ["a", "b", "label"]
    assert len(dataset.df) == 3
