import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from skore.sklearn._base import _get_cached_response_values


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
        "data_source_hash": 456,
    }
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    initial_calls = getattr(estimator, f"n_call_{response_method}")

    # Second call with different hash should trigger new computation
    params["data_source_hash"] = 789
    response_values = _get_cached_response_values(cache=cache, **params)
    assert response_values.shape == y.shape
    current_calls = getattr(estimator, f"n_call_{response_method}")
    assert current_calls == initial_calls + 1, (
        f"Different data source hash should trigger new "
        f"computation for {response_method}"
    )
