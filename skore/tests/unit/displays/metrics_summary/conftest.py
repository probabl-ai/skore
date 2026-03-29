"""Fixtures for metrics summary display tests."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data


class CustomClassifierWithoutPredictProba(ClassifierMixin, BaseEstimator):
    """Binary classifier with only `predict` (no `predict_proba`), mirroring the
    sklearn-api integration example.
    """

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


@pytest.fixture
def custom_classifier_no_predict_proba_with_test(
    binary_classification_train_test_split,
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    estimator = CustomClassifierWithoutPredictProba().fit(X_train, y_train)
    return estimator, X_test, y_test


@pytest.fixture
def custom_classifier_no_predict_proba_data(binary_classification_data):
    X, y = binary_classification_data
    return CustomClassifierWithoutPredictProba(), X, y
