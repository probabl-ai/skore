import pytest
from skore import EstimatorReport


def test_wrong_report_type(pyplot, logistic_binary_classification_with_train_test):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `report_type` argument."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.roc()
    display.report_type = "unknown"
    err_msg = (
        "`report_type` should be one of 'estimator', 'cross-validation', "
        "'comparison-cross-validation' or 'comparison-estimator'. "
        "Got 'unknown' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot()
