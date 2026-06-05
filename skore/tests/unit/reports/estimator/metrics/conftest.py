import pytest

from skore._sklearn._estimator.report import EstimatorReport


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
def regression_report(linear_regression_with_train_test):
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
