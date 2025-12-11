import pytest

from skore import EstimatorReport


def test_wrong_subplot_by(pyplot, forest_binary_classification_with_train_test):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.precision_recall()
    err_msg = "subplot_by must be one of"
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")
