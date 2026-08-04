"""Shared fixtures for report unit tests."""

import pytest

from skore import EstimatorReport


@pytest.fixture
def binary_classification_report(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label=1,
    )


@pytest.fixture
def multiclass_classification_report(
    logistic_multiclass_classification_with_train_test,
):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def regression_report(linear_regression_with_train_test):
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def multioutput_regression_report(linear_regression_multioutput_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def svc_binary_classification_report(svc_binary_classification_with_train_test):
    """SVC binary report: has decision_function but no predict_proba."""
    estimator, X_train, X_test, y_train, y_test = (
        svc_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def classifier_no_predict_proba_report(
    custom_classifier_no_predict_proba_with_test,
):
    """Custom classifier without predict_proba and decision_function."""
    estimator, X_test, y_test = custom_classifier_no_predict_proba_with_test
    return EstimatorReport(estimator, X_test=X_test, y_test=y_test)
