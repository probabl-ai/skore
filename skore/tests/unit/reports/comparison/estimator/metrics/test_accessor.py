import pytest
from sklearn.dummy import DummyClassifier

from skore import ComparisonReport, EstimatorReport


def test_report_without_testing_data(binary_classification_data):
    """If there is no test data (`None`) for some estimator report,
    initialization works, but computing metrics can fail.
    """
    X_train, y_train = binary_classification_data
    estimator_report_1 = EstimatorReport(
        DummyClassifier(), X_train=X_train, y_train=y_train
    )
    estimator_report_2 = EstimatorReport(
        DummyClassifier(), X_train=X_train, y_train=y_train
    )

    report = ComparisonReport([estimator_report_1, estimator_report_2])

    with pytest.raises(ValueError, match="No test data"):
        report.metrics.summarize(data_source="test")


def test_random_state(comparison_estimator_reports_regression):
    """If random_state is None (the default) the call should not be cached."""
    report = comparison_estimator_reports_regression
    report.metrics.prediction_error()
    # skore should store the y_pred of the internal estimators, but not the plot
    assert report._cache == {}
