import numpy as np
import pytest
from sklearn.cluster import KMeans

from skore import ComparisonReport, CrossValidationReport


def test_init_wrong_parameters(cross_validation_reports_binary_classification):
    """If the input is not valid, raise."""
    report, _ = cross_validation_reports_binary_classification

    with pytest.raises(TypeError, match="Expected reports to be a list or dict"):
        ComparisonReport(report)

    with pytest.raises(ValueError, match="Expected at least 2 reports to compare"):
        ComparisonReport([report])

    with pytest.raises(
        TypeError, match="Expected .* EstimatorReport or .* CrossValidationReport"
    ):
        ComparisonReport([None, report])


def test_init_different_ml_usecases(
    cross_validation_reports_binary_classification, cross_validation_reports_regression
):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        regression_report, _ = cross_validation_reports_regression
        classification_report, _ = cross_validation_reports_binary_classification
        ComparisonReport([regression_report, classification_report])


def test_init_non_distinct_reports(cross_validation_reports_binary_classification):
    """If the passed estimators are not distinct objects, raise an error."""
    same_report, _ = cross_validation_reports_binary_classification

    with pytest.raises(ValueError, match="Expected reports to be distinct objects"):
        ComparisonReport([same_report, same_report])


def test_non_string_report_names(cross_validation_reports_binary_classification):
    """If the estimators are passed as a dict, then keys must be strings."""
    cv_report_1, cv_report_2 = cross_validation_reports_binary_classification
    with pytest.raises(TypeError, match="Expected all report names to be strings"):
        ComparisonReport({0: cv_report_1, "1": cv_report_2})


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_get_predictions(
    comparison_cross_validation_reports_binary_classification,
    binary_classification_data,
    data_source,
):
    report = comparison_cross_validation_reports_binary_classification
    if data_source == "X_y":
        X, _ = binary_classification_data
        predictions = report.get_predictions(X=X, data_source=data_source)

    else:
        predictions = report.get_predictions(data_source=data_source)

    assert len(predictions) == len(report.reports_)
    for i, cv_report in enumerate(report.reports_.values()):
        assert len(predictions[i]) == cv_report._splitter.n_splits


def test_clustering():
    """Check that we cannot create a report with a clustering model."""
    with pytest.raises(
        ValueError,
        match="skore does not support clustering models yet. Please use a "
        "classification or regression model instead.",
    ):
        ComparisonReport(
            [
                CrossValidationReport(KMeans(), X=np.random.rand(10, 5)),
                CrossValidationReport(KMeans(), X=np.random.rand(10, 5)),
            ]
        )
