import pytest
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from skore import EstimatorReport


def test_estimator_not_fitted():
    estimator = LinearRegression()
    with pytest.raises(NotFittedError):
        EstimatorReport.from_fitted_estimator(estimator, X=None, y=None)


def test_estimator_report_from_unfitted_estimator():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LinearRegression()
    report = EstimatorReport.from_unfitted_estimator(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    check_is_fitted(report.estimator)
    assert report.estimator is not estimator  # the estimator should be cloned

    assert report.X_train is X_train
    assert report.y_train is y_train
    assert report.X_val is X_test
    assert report.y_val is y_test


def test_estimator_report_from_fitted_estimator():
    X, y = make_regression(random_state=42)
    estimator = LinearRegression().fit(X, y)
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    assert report.estimator is estimator  # we should not clone the estimator
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_val is X
    assert report.y_val is y
