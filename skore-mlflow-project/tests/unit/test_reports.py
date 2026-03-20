import numpy as np
import pandas as pd
import pytest

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
    X = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "c"], dtype="category"),
            "num": [1, 2, 3],
        }
    )

    sample = _sample_input_example(X, max_samples=2)

    assert sample.shape == (2, 2)
    assert not isinstance(sample["cat"].dtype, pd.CategoricalDtype)
    assert sample["cat"].tolist() == ["a", "b"]


def test_dataset_from_Xy_accepts_dataframe_X_and_numpy_y() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = np.array(["no cancer", "cancer", "no cancer"], dtype=object)

    dataset = _dataset_from_Xy(X, y)

    assert type(dataset.dataset).__name__ == "PandasDataset"
    assert dataset.dataset.targets == "target"


def test_dataset_from_Xy_accepts_numpy_X_and_series_y() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = pd.Series(["no cancer", "cancer", "no cancer"], name="label")

    dataset = _dataset_from_Xy(X, y)

    assert type(dataset.dataset).__name__ == "NumpyDataset"
    assert isinstance(dataset.dataset.targets, np.ndarray)
    assert dataset.dataset.targets.tolist() == y.tolist()


def test_dataset_from_Xy_accepts_numpy_X_and_dataframe_y() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = pd.DataFrame(
        {
            "label": ["no cancer", "cancer", "no cancer"],
            "weight": [0.1, 0.9, 0.2],
        }
    )

    dataset = _dataset_from_Xy(X, y)

    assert type(dataset.dataset).__name__ == "NumpyDataset"
    assert isinstance(dataset.dataset.targets, dict)
    assert dataset.dataset.targets["label"].tolist() == y["label"].tolist()
    assert dataset.dataset.targets["weight"].tolist() == y["weight"].tolist()
