import pandas as pd
import pytest
from skore import ComparisonReport, EstimatorReport


def test_init_wrong_parameters(
    estimator_reports_binary_classification,
):
    """If the input is not valid, raise."""
    report, _ = estimator_reports_binary_classification
    with pytest.raises(TypeError, match="Expected reports to be a list or dict"):
        ComparisonReport(report)

    with pytest.raises(ValueError, match="Expected at least 2 reports to compare"):
        ComparisonReport([report])

    with pytest.raises(
        TypeError, match="Expected .* EstimatorReport or .* CrossValidationReport"
    ):
        ComparisonReport([None, report])


def test_different_test_data(
    logistic_binary_classification_with_train_test,
):
    """Raise an error if the passed estimators do not have the same testing targets."""
    estimator, _, X_test, _, y_test = logistic_binary_classification_with_train_test

    # The estimators that have testing data need to have the same testing targets
    with pytest.raises(
        ValueError, match="Expected all estimators to share the same test targets."
    ):
        ComparisonReport(
            [
                EstimatorReport(estimator, y_test=y_test),
                EstimatorReport(estimator, y_test=y_test[1:]),
            ]
        )

    # The estimators without testing data (i.e., no y_test) do not count
    ComparisonReport(
        [
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator, X_test=X_test, y_test=y_test),
            EstimatorReport(estimator),
        ]
    )

    # If there is an X_test but no y_test, it should not raise an error
    ComparisonReport(
        [
            EstimatorReport(estimator, fit=False, X_test=X_test, y_test=None),
            EstimatorReport(estimator, fit=False, X_test=X_test, y_test=None),
        ]
    )


def test_init_different_ml_usecases(
    estimator_reports_binary_classification, estimator_reports_regression
):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    classification_report, _ = estimator_reports_binary_classification
    regression_report, _ = estimator_reports_regression

    # Simulate that the regression and classification reports share the same testing
    # targets which is not unlikely.
    regression_report._y_test = classification_report._y_test
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        ComparisonReport([classification_report, regression_report])


def test_init_with_report_names(
    estimator_reports_binary_classification,
):
    """If the estimators are passed as a dict then the estimator names are the dict
    keys."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification

    report = ComparisonReport({"r1": estimator_report_1, "r2": estimator_report_2})

    pd.testing.assert_index_equal(
        report.metrics.accuracy().columns,
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_init_without_report_names(
    estimator_reports_binary_classification,
):
    """If the estimators are passed as a list, then the estimator names are the
    estimator class names."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification

    report = ComparisonReport([estimator_report_1, estimator_report_2])

    pd.testing.assert_index_equal(
        report.metrics.accuracy().columns,
        pd.Index(["DummyClassifier_1", "DummyClassifier_2"], name="Estimator"),
    )


def test_non_string_report_names(
    estimator_reports_binary_classification,
):
    """If the reports are passed as a dict, then keys must be strings."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    with pytest.raises(TypeError, match="Expected all report names to be strings"):
        ComparisonReport({0: estimator_report_1, "1": estimator_report_2})


@pytest.mark.parametrize("data_source", ["train", "test"])
@pytest.mark.parametrize("response_method", ["predict", "predict_proba"])
@pytest.mark.parametrize("pos_label", [None, 0, 1])
def test_get_predictions(
    comparison_estimator_reports_binary_classification,
    data_source,
    response_method,
    pos_label,
):
    """Check the behaviour of the `get_predictions` method."""
    report = comparison_estimator_reports_binary_classification
    predictions = report.get_predictions(
        data_source=data_source, response_method=response_method, pos_label=pos_label
    )
    assert len(predictions) == 2
    sub_reports = list(report.reports_.values())
    for split_idx, split_predictions in enumerate(predictions):
        if data_source == "train":
            expected_shape = sub_reports[split_idx].y_train.shape
        else:
            expected_shape = sub_reports[split_idx].y_test.shape
        assert split_predictions.shape == expected_shape


def test_get_predictions_error(
    comparison_estimator_reports_binary_classification,
):
    """Check that we raise an error when the data source is invalid."""
    report = comparison_estimator_reports_binary_classification
    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid")
