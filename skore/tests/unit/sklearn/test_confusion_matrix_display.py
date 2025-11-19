import numpy as np
import pytest

from skore._sklearn._plot.metrics.confusion_matrix import ConfusionMatrixDisplay
from skore._sklearn.types import YPlotData


# -------------------------------------------------------------------
# Helper class for mocking Estimator-like objects with a `.name` attr
# -------------------------------------------------------------------
class DummyEstimator:
    def __init__(self, name):
        self.name = name


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture
def sample_binary_data():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    return y_true, y_pred


@pytest.fixture
def sample_cv_data():
    # 3 folds
    y_trues = [
        np.array([0, 1, 0]),
        np.array([1, 1, 0]),
        np.array([0, 0, 1]),
    ]
    y_preds = [
        np.array([0, 1, 1]),
        np.array([1, 0, 0]),
        np.array([0, 1, 1]),
    ]
    return y_trues, y_preds


# -------------------------------------------------------------------
# Test: Estimator confusion matrix
# -------------------------------------------------------------------
def test_single_estimator(sample_binary_data):
    y_true, y_pred = sample_binary_data

    disp = ConfusionMatrixDisplay._compute_data_for_display(
        y_true=[YPlotData(y_true)],
        y_pred=[YPlotData(y_pred)],
        report_type="estimator",
        display_labels=None,
        normalize=None,
    )

    assert disp.report_type == "estimator"
    assert disp.confusion_matrix.shape == (2, 2)

    # Should not raise
    disp.plot()


# -------------------------------------------------------------------
# Test: Cross-validation confusion matrix (mean ± std)
# -------------------------------------------------------------------
def test_cross_validation(sample_cv_data):
    y_trues, y_preds = sample_cv_data

    disp = ConfusionMatrixDisplay._compute_data_for_display(
        y_true=[YPlotData(y) for y in y_trues],
        y_pred=[YPlotData(y) for y in y_preds],
        report_type="cross-validation",
        display_labels=None,
        normalize=None,
    )

    cms = disp.confusion_matrix
    assert isinstance(cms, list)
    assert len(cms) == 3
    assert cms[0].shape == (2, 2)

    # Should compute mean ± std without error
    disp.plot()


# -------------------------------------------------------------------
# Test: Comparison report → multiple subplots
# -------------------------------------------------------------------
def test_comparison_report(sample_binary_data):
    y_true, y_pred = sample_binary_data

    estimators = [DummyEstimator("ModelA"), DummyEstimator("ModelB")]

    disp = ConfusionMatrixDisplay._compute_data_for_display(
        y_true=[YPlotData(y_true), YPlotData(y_true)],
        y_pred=[YPlotData(y_pred), YPlotData(y_pred)],
        report_type="comparison-estimator",
        display_labels=None,
        estimators=estimators,
        normalize=None,
    )

    assert isinstance(disp.confusion_matrix, dict)
    assert "ModelA" in disp.confusion_matrix
    assert "ModelB" in disp.confusion_matrix
    assert disp.confusion_matrix["ModelA"].shape == (2, 2)

    # Should create subplots without error
    disp.plot()


# -------------------------------------------------------------------
# Test: display label inference
# -------------------------------------------------------------------
def test_display_labels_inference(sample_binary_data):
    y_true, y_pred = sample_binary_data

    disp = ConfusionMatrixDisplay._compute_data_for_display(
        y_true=[YPlotData(y_true)],
        y_pred=[YPlotData(y_pred)],
        report_type="estimator",
        display_labels=None,
        normalize=None,
    )

    assert disp.display_labels == ["0", "1"]  # inferred labels


# -------------------------------------------------------------------
# Test: Incorrect label length throws an error
# -------------------------------------------------------------------
def test_display_labels_wrong_length(sample_binary_data):
    y_true, y_pred = sample_binary_data

    with pytest.raises(ValueError):
        ConfusionMatrixDisplay._compute_data_for_display(
            y_true=[YPlotData(y_true)],
            y_pred=[YPlotData(y_pred)],
            report_type="estimator",
            display_labels=["A"],  # wrong length
            normalize=None,
        )
