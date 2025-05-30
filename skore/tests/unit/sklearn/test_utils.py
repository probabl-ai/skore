import numpy
import pandas
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from skore.sklearn._plot.utils import _ClassifierCurveDisplayMixin
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
        (
            *make_regression(n_targets=2, random_state=42),
            LinearRegression(),
            "multioutput-regression",
        ),
        (make_classification(random_state=42)[0], None, KMeans(), "clustering"),
        (
            *make_multilabel_classification(random_state=42),
            MultiOutputClassifier(LogisticRegression()),
            "multioutput-binary-classification",
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


def test_find_ml_task_with_estimator_multiclass():
    estimator = LogisticRegression().fit(
        *make_classification(n_classes=3, n_informative=3, random_state=42)
    )
    assert _find_ml_task(None, estimator) == "multiclass-classification"


@pytest.mark.parametrize(
    "target, expected_task",
    [
        (make_classification(random_state=42)[1], "binary-classification"),
        (
            make_classification(
                n_classes=3,
                n_clusters_per_class=1,
                random_state=42,
            )[1],
            "multiclass-classification",
        ),
        (make_regression(n_samples=100, random_state=42)[1], "regression"),
        (None, "clustering"),
        (
            make_multilabel_classification(random_state=42)[1],
            "multioutput-binary-classification",
        ),
        (
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            "multioutput-multiclass-classification",
        ),
        (numpy.array([1, 5, 9]), "regression"),
        (numpy.array([0, 1, 2]), "multiclass-classification"),
        (numpy.array([1, 2, 3]), "regression"),
        (numpy.array([0, 1, 5, 9]), "regression"),
        # Non-integer target
        (numpy.array([[0.5, 2]]), "multioutput-regression"),
        # No 0 class
        (numpy.array([[1, 2], [2, 1]]), "multioutput-regression"),
        # No 2 class
        (numpy.array([[0, 3], [1, 3]]), "multioutput-regression"),
        # Non-numeric is classification
        (numpy.array(["a", "b", "c"]), "multiclass-classification"),
        (numpy.array([[0, 2], [1, 2]]), "multioutput-regression"),
        (numpy.array([[[0], [0]]]), "unknown"),
    ],
)
def test_find_ml_task_without_estimator(target, expected_task):
    assert _find_ml_task(target) == expected_task


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (DummyClassifier(), "classification"),
        (DummyRegressor(), "regression"),
    ],
)
def test_find_ml_task_unfitted_estimator(estimator, expected):
    assert _find_ml_task(None, estimator) == expected


def test_find_ml_task_pandas():
    y = pandas.Series([0, 1, 2])
    assert _find_ml_task(y, None) == "multiclass-classification"

    y = pandas.DataFrame([0, 1, 2])
    assert _find_ml_task(y, None) == "multiclass-classification"


def test_find_ml_task_string():
    assert _find_ml_task(["0", "1", "2"], None) == "multiclass-classification"


class Test_ClassifierCurveDisplayMixin:
    def test__threshold_average(self):
        xs = [numpy.array([3, 2, 1]), numpy.array([3, 2, 1])]
        ys = [numpy.array([3, 2, 1]), numpy.array([3, 2, 1])]
        thresholds = [numpy.array([4, 3, 1]), numpy.array([5, 3, 2])]
        x, y, threshold = _ClassifierCurveDisplayMixin._threshold_average(
            xs, ys, thresholds
        )
        expected = numpy.array([3, 2.5, 2, 1, 1])
        assert_array_equal(x, expected)
        assert_array_equal(y, expected)
        assert_array_equal(threshold, numpy.array([5, 4, 3, 2, 1]))
