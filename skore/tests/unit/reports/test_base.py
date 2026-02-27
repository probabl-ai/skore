import re

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore._sklearn._base import _get_cached_response_values
from skore._utils._testing import MockAccessor, MockReport


class MockClassifier(ClassifierMixin, BaseEstimator):
    """Minimal scikit-learn compatible classifier."""

    def __init__(
        self, n_call_predict=0, n_call_predict_proba=0, n_call_decision_function=0
    ):
        self.n_call_predict = n_call_predict
        self.n_call_predict_proba = n_call_predict_proba
        self.n_call_decision_function = n_call_decision_function

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fitted_ = True
        return self

    def predict(self, X):
        self.n_call_predict += 1
        return np.ones(X.shape[0])

    def predict_proba(self, X):
        self.n_call_predict_proba += 1
        return np.ones((X.shape[0], len(self.classes_)))

    def decision_function(self, X):
        self.n_call_decision_function += 1
        return np.ones(X.shape[0])


class MockRegressor(RegressorMixin, BaseEstimator):
    """Minimal scikit-learn compatible regressor."""

    def __init__(self, n_call_predict=0):
        self.n_call_predict = n_call_predict

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        self.n_call_predict += 1
        return np.ones(X.shape[0])


@pytest.mark.parametrize(
    "Estimator, response_method, pos_label_sensitive",
    [
        (MockClassifier, "predict", False),
        (MockClassifier, "predict_proba", True),
        (MockClassifier, "decision_function", True),
        (MockRegressor, "predict", False),
    ],
)
@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize("pos_label", [0, 1])
def test_get_cached_response_values(
    Estimator,
    response_method,
    pos_label_sensitive,
    data_source,
    pos_label,
):
    """Check the general behavior of the `_get_cached_response_values` function."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    cache = {}
    estimator = Estimator().fit(X, y)

    params = {
        "estimator_hash": 123,
        "estimator": estimator,
        "X": X,
        "response_method": response_method,
        "pos_label": pos_label,
        "data_source": data_source,
    }

    # trigger the computation
    results = _get_cached_response_values(cache=cache, **params)

    # when the predictions are not cached then we have 2 results:
    # - the prediction
    # - the time it took to compute the prediction
    assert len(results) == 2
    assert results[0][1].shape == y.shape  # check the predictions shape
    initial_calls = getattr(estimator, f"n_call_{response_method}")
    assert initial_calls == 1, (
        f"Expected 1 call for {response_method}, got {initial_calls}"
    )

    # cache the results
    cache.update((key, value) for key, value, _ in results)

    # Reload from the cache
    results = _get_cached_response_values(cache=cache, **params)
    assert len(results) == 1
    _, response_values, is_cached = results[0]
    assert is_cached
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert current_calls == initial_calls, (
        f"Cache was not used for {response_method} "
        f"(calls: {current_calls} != {initial_calls})"
    )

    # Change pos_label
    # It should trigger recomputation for predict_proba and decision_function only
    params["pos_label"] = 1 - pos_label
    results = _get_cached_response_values(cache=cache, **params)
    current_calls = getattr(estimator, f"n_call_{response_method}")

    if pos_label_sensitive:
        assert len(results) == 2
        assert current_calls == initial_calls + 1
        _, response_values, is_cached = results[0]
        assert not is_cached
        assert response_values.shape == y.shape

        cache.update((key, value) for key, value, _ in results)

    else:
        assert len(results) == 1
        assert current_calls == initial_calls
        _, response_values, is_cached = results[0]
        assert is_cached
        assert response_values.shape == y.shape

    # Should reload completely from the cache
    results = _get_cached_response_values(cache=cache, **params)
    assert len(results) == 1
    _, response_values, is_cached = results[0]
    assert is_cached
    assert response_values.shape == y.shape


def test_base_accessor_get_X_y_and_data_source_hash_error():
    """Check that we raise the proper error in `get_X_y_and_use_cache`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = MockReport(estimator, X_train=None, y_train=None, X_test=None, y_test=None)
    accessor = MockAccessor(parent=report)

    err_msg = re.escape(
        "Invalid data source: unknown. Possible values are: test, train."
    )
    with pytest.raises(ValueError, match=err_msg):
        accessor._get_X_y_and_data_source_hash(data_source="unknown")

    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source} and y_{data_source}) were "
            f"provided when creating the report. Please provide the {data_source} "
            "data when creating the report."
        )
        with pytest.raises(ValueError, match=err_msg):
            accessor._get_X_y_and_data_source_hash(data_source=data_source)

    report = MockReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    accessor = MockAccessor(parent=report)


@pytest.mark.parametrize("data_source", ("train", "test"))
def test_base_accessor_get_X_y_and_data_source_hash(data_source):
    """Check the general behaviour of `get_X_y_and_use_cache`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = MockReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    accessor = MockAccessor(parent=report)
    X, y = accessor._get_X_y_and_data_source_hash(data_source=data_source)

    if data_source == "train":
        assert X is X_train
        assert y is y_train
    else:
        assert data_source == "test"
        assert X is X_test
        assert y is y_test
