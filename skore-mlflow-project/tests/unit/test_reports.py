from itertools import product

import numpy as np
import pandas as pd
import pytest
from mlflow.data.numpy_dataset import NumpyDataset
from mlflow.data.pandas_dataset import PandasDataset
from numpy.testing import assert_array_equal

from skore_mlflow_project.reports import (
    Artifact,
    Metric,
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


@pytest.mark.parametrize(
    ("X", "y"),
    product([X_pandas, X_numpy], [y_pandas, y_numpy, y_pandas_multi_targets]),
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
