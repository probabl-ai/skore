import re

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport


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


# TODO: replace those with tests on ._get_predictions(...)?


def test_report_get_X_y_error():
    """Check that we raise the proper error in `_get_X_y`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape(
        "Invalid data source: unknown. Possible values are: test, train."
    )
    with pytest.raises(ValueError, match=err_msg):
        report._get_X_y(data_source="unknown")

    err_msg = re.escape(
        "No train data (i.e. X_train and y_train) were provided "
        "when creating the report. Please provide the train "
        "data when creating the report."
    )
    with pytest.raises(ValueError, match=err_msg):
        report._get_X_y(data_source="train")


@pytest.mark.parametrize("data_source", ("train", "test"))
def test_report_get_X_y(data_source):
    """Check the general behaviour of `_get_X_y`."""
    X, y = make_classification(n_samples=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    X_result, y_result = report._get_X_y(data_source=data_source)

    if data_source == "train":
        np.testing.assert_array_equal(X_result, X_train)
        np.testing.assert_array_equal(y_result, y_train)
    else:
        assert data_source == "test"
        np.testing.assert_array_equal(X_result, X_test)
        np.testing.assert_array_equal(y_result, y_test)
