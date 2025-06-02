import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport


@pytest.fixture
def cv_report_classification(classification_data):
    X, y = classification_data
    cv_report = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=0), X, y
    )
    return cv_report


@pytest.fixture
def cv_report_regression():
    X, y = make_regression(random_state=42)
    cv_report = CrossValidationReport(DummyRegressor(), X, y)
    return cv_report


def test_init_wrong_parameters(cv_report_classification):
    """If the input is not valid, raise."""

    with pytest.raises(TypeError, match="Expected reports to be a list or dict"):
        ComparisonReport(cv_report_classification)

    with pytest.raises(ValueError, match="Expected at least 2 reports to compare"):
        ComparisonReport([cv_report_classification])

    with pytest.raises(
        TypeError, match="Expected .* EstimatorReport or .* CrossValidationReport"
    ):
        ComparisonReport([None, cv_report_classification])


def test_init_different_ml_usecases(cv_report_classification, cv_report_regression):
    """Raise an error if the passed estimators do not have the same ML usecase."""
    with pytest.raises(
        ValueError, match="Expected all estimators to have the same ML usecase"
    ):
        ComparisonReport([cv_report_regression, cv_report_classification])


def test_init_non_distinct_reports(cv_report_classification):
    """If the passed estimators are not distinct objects, raise an error."""

    with pytest.raises(ValueError, match="Expected reports to be distinct objects"):
        ComparisonReport([cv_report_classification, cv_report_classification])


def test_non_string_report_names(cv_reports):
    """If the estimators are passed as a dict, then keys must be strings."""
    cv_report_1, cv_report_2 = cv_reports
    with pytest.raises(TypeError, match="Expected all report names to be strings"):
        ComparisonReport({0: cv_report_1, "1": cv_report_2})


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_get_predictions(report_cv_reports, classification_data, data_source):
    if data_source == "X_y":
        X, _ = classification_data
        predictions = report_cv_reports.get_predictions(X=X, data_source=data_source)

    else:
        predictions = report_cv_reports.get_predictions(data_source=data_source)

    assert len(predictions) == len(report_cv_reports.reports_)
    for i, cv_report in enumerate(report_cv_reports.reports_):
        assert len(predictions[i]) == cv_report._cv_splitter.n_splits


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_comparison_report_display_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y)
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y)
    report = ComparisonReport([report_1, report_2])
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y, pos_label="A")
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y, pos_label="A")
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()
