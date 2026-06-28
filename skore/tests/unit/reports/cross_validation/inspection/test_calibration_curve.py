import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from skore import CalibrationDisplay, CrossValidationReport


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
    ],
)
def test_with_model_exposing_calibration_curve(binary_classification_data, estimator):
    """Check that we can create a calibration curve display from supported models."""
    X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    assert hasattr(report.inspection, "calibration_curve")
    display = report.inspection.calibration_curve()

    assert isinstance(display, CalibrationDisplay)
    assert set(display.calibration_report["split"]) == {0, 1}


def test_with_regressor(binary_classification_data):
    """Check that regressors do not expose calibration curve."""
    X, y = binary_classification_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)

    assert not hasattr(report.inspection, "calibration_curve")


def test_without_predict_proba(binary_classification_data):
    """Check that classifiers without `predict_proba` do not expose calibration."""
    X, y = binary_classification_data
    report = CrossValidationReport(LinearSVC(random_state=0), X, y, splitter=2)

    assert not hasattr(report.inspection, "calibration_curve")


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(binary_classification_data, data_source):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=2)

    display = report.inspection.calibration_curve(data_source=data_source)

    assert set(display.calibration_report["data_source"]) == {data_source}


def test_frame_default_aggregate(binary_classification_data):
    X, y = binary_classification_data
    report = CrossValidationReport(LogisticRegression(), X, y, splitter=2)

    frame = report.inspection.calibration_curve().frame()

    assert "split" not in frame.columns
    assert "predicted_probability_mean" in frame.columns
    assert "fraction_of_positives_std" in frame.columns
