import numpy
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from skore.sklearn.find_ml_task import _find_ml_task


@pytest.mark.parametrize(
    "X, y, estimator, expected_task",
    [
        (
            *make_classification(random_state=42),
            LogisticRegression(),
            "binary-classification",
        ),
        (
            *make_regression(random_state=42),
            LinearRegression(),
            "regression",
        ),
        (make_classification(random_state=42)[0], None, KMeans(), "clustering"),
        (
            *make_multilabel_classification(random_state=42),
            MultiOutputClassifier(LogisticRegression()),
            "unknown",
        ),
    ],
)
@pytest.mark.parametrize("should_fit", [True, False])
def test_find_ml_task_with_estimator(X, y, estimator, expected_task, should_fit):
    if isinstance(estimator, KMeans):
        y = None
    if should_fit:
        estimator.fit(X, y)
    assert _find_ml_task(y, estimator) == expected_task


@pytest.mark.parametrize(
    "target, expected_task",
    [
        (make_classification(random_state=42)[1], "binary-classification"),
        (
            make_classification(n_classes=3, n_clusters_per_class=1, random_state=42)[
                1
            ],
            "multiclass-classification",
        ),
        (make_regression(n_samples=100, random_state=42)[1], "regression"),
        (None, "clustering"),
        (make_multilabel_classification(random_state=42)[1], "unknown"),
        (numpy.array([1, 5, 9]), "regression"),
        (numpy.array([0, 1, 2]), "multiclass-classification"),
        (numpy.array([1, 2, 3]), "regression"),
        (numpy.array([0, 1, 5, 9]), "regression"),
    ],
)
def test_find_ml_task_without_estimator(target, expected_task):
    assert _find_ml_task(target) == expected_task


def test_find_ml_task_unfitted_estimator():
    from sklearn.dummy import DummyClassifier

    estimator = DummyClassifier()
    assert _find_ml_task(None, estimator) == "unknown"
