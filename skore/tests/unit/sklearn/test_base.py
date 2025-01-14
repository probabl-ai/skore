import re

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore.sklearn._base import _BaseAccessor, _get_cached_response_values


class MockClassifier(ClassifierMixin, BaseEstimator):
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
@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
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
        "data_source_hash": None if data_source == "test" else 456,
    }

    # trigger the computation
    response_values = _get_cached_response_values(cache=cache, **params)

    assert response_values.shape == y.shape
    initial_calls = getattr(estimator, f"n_call_{response_method}")
    assert (
        initial_calls == 1
    ), f"Expected 1 call for {response_method}, got {initial_calls}"

    # Reload from the cache
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert current_calls == initial_calls, (
        f"Cache was not used for {response_method} "
        f"(calls: {current_calls} != {initial_calls})"
    )

    # Change pos_label
    # It should trigger recomputation for predict_proba and decision_function only
    params["pos_label"] = 1 - pos_label
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    expected_calls = initial_calls + (1 if pos_label_sensitive else 0)
    assert (
        current_calls == expected_calls
    ), f"Unexpected number of calls for different pos_label in {response_method}"

    # Should reload completely from the cache
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert (
        current_calls == expected_calls
    ), f"Unexpected number of calls for different pos_label in {response_method}"


@pytest.mark.parametrize(
    "Estimator, response_method",
    [
        (MockClassifier, "predict"),
        (MockClassifier, "predict_proba"),
        (MockClassifier, "decision_function"),
        (MockRegressor, "predict"),
    ],
)
def test_get_cached_response_values_different_data_source_hash(
    Estimator, response_method
):
    """Test that different data source hashes trigger new computations."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    cache = {}
    estimator = Estimator().fit(X, y)

    params = {
        "estimator_hash": 123,
        "estimator": estimator,
        "X": X,
        "response_method": response_method,
        "pos_label": 1,
        "data_source": "X_y",
        "data_source_hash": None,
    }
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    initial_calls = getattr(estimator, f"n_call_{response_method}")

    # Second call by passing the hash of the data should not trigger new computation
    # because we consider it trustworthy
    params["data_source_hash"] = joblib.hash(X)
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert current_calls == initial_calls, (
        f"Passing a hash corresponding to the data should not trigger new "
        f"computation for {response_method}"
    )

    # Third call by passing a data hash not in the keys should trigger new computation
    # It is should never happen in practice but the behaviour is safe
    params["data_source_hash"] = 456
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert current_calls == initial_calls + 1, (
        f"Passing a hash not present in the cache keys should trigger new "
        f"computation for {response_method}"
    )


class MockReport:
    def __init__(self, estimator, X_train=None, y_train=None, X_test=None, y_test=None):
        """Mock a reporter with the minimal required attributes."""
        self._estimator = estimator
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

    @property
    def estimator(self):
        return self._estimator

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_test(self):
        return self._y_test


def test_base_accessor_get_X_y_and_data_source_hash_error():
    """Check that we raise the proper error in `get_X_y_and_use_cache`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = MockReport(estimator, X_train=None, y_train=None, X_test=None, y_test=None)
    accessor = _BaseAccessor(parent=report, icon="")

    err_msg = re.escape(
        "Invalid data source: unknown. Possible values are: " "test, train, X_y."
    )
    with pytest.raises(ValueError, match=err_msg):
        accessor._get_X_y_and_data_source_hash(data_source="unknown")

    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source} and y_{data_source}) were "
            f"provided when creating the reporter. Please provide the {data_source} "
            "data either when creating the reporter or by setting data_source to "
            "'X_y' and providing X and y."
        )
        with pytest.raises(ValueError, match=err_msg):
            accessor._get_X_y_and_data_source_hash(data_source=data_source)

    report = MockReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    accessor = _BaseAccessor(parent=report, icon="")

    for data_source in ("train", "test"):
        err_msg = f"X and y must be None when data_source is {data_source}."
        with pytest.raises(ValueError, match=err_msg):
            accessor._get_X_y_and_data_source_hash(
                data_source=data_source, X=X_test, y=y_test
            )

    err_msg = "X and y must be provided."
    with pytest.raises(ValueError, match=err_msg):
        accessor._get_X_y_and_data_source_hash(data_source="X_y")

    # FIXME: once we choose some basic metrics for clustering, then we don't need to
    # use `custom_metric` for them.
    estimator = KMeans(n_clusters=2).fit(X_train)
    report = MockReport(estimator, X_test=X_test)
    accessor = _BaseAccessor(parent=report, icon="")
    err_msg = "X must be provided."
    with pytest.raises(ValueError, match=err_msg):
        accessor._get_X_y_and_data_source_hash(data_source="X_y")

    report = MockReport(estimator)
    accessor = _BaseAccessor(parent=report, icon="")
    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source}) were provided when "
            f"creating the reporter. Please provide the {data_source} data either "
            f"when creating the reporter or by setting data_source to 'X_y' and "
            f"providing X and y."
        )
        with pytest.raises(ValueError, match=err_msg):
            accessor._get_X_y_and_data_source_hash(data_source=data_source)


@pytest.mark.parametrize("data_source", ("train", "test", "X_y"))
def test_base_accessor_get_X_y_and_data_source_hash(data_source):
    """Check the general behaviour of `get_X_y_and_use_cache`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = MockReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    accessor = _BaseAccessor(parent=report, icon="")
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    X, y, data_source_hash = accessor._get_X_y_and_data_source_hash(
        data_source=data_source, **kwargs
    )

    if data_source == "train":
        assert X is X_train
        assert y is y_train
        assert data_source_hash is None
    elif data_source == "test":
        assert X is X_test
        assert y is y_test
        assert data_source_hash is None
    elif data_source == "X_y":
        assert X is X_test
        assert y is y_test
        assert data_source_hash == joblib.hash((X_test, y_test))
