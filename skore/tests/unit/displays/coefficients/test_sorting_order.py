import numpy as np
from sklearn.linear_model import Ridge

from skore import ComparisonReport, EstimatorReport


def test_descending(regression_train_test_split):
    """Test that sorting_order='descending' sorts per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="descending")

    # For each estimator, verify sorting
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        abs_coefs = est_frame["coefficients"].abs().values
        assert np.all(abs_coefs[:-1] >= abs_coefs[1:]), (
            f"Features not sorted in descending order for {estimator_name}"
        )


def test_ascending(regression_train_test_split):
    """Test that sorting_order='ascending' sorts per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="ascending")

    # For each estimator, verify sorting
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        abs_coefs = est_frame["coefficients"].abs().values
        assert np.all(abs_coefs[:-1] <= abs_coefs[1:]), (
            f"Features not sorted in ascending order for {estimator_name}"
        )


def test_plot(regression_train_test_split):
    """Test that plot method correctly uses sorting_order parameter."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    # Should not raise an error
    display.plot(sorting_order="ascending", include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None
