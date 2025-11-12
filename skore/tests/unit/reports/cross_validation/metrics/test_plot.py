import numpy as np
import pytest

from skore import CrossValidationReport, RocCurveDisplay


def test_plot_roc(forest_binary_classification_data):
    """Check that the ROC plot method works."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert isinstance(report.metrics.roc(), RocCurveDisplay)


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_display_binary_classification(
    pyplot, forest_binary_classification_data, display
):
    """General behaviour of the function creating display on binary classification."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)()
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)()
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_display_regression(pyplot, linear_regression_data, display):
    """General behaviour of the function creating display on regression."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(seed=0)
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(seed=0)
    assert display_first_call is display_second_call


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_pos_label(
    pyplot, metric, forest_binary_classification_data
):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    classifier, X, y = forest_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report = CrossValidationReport(classifier, X, y)
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

    report = CrossValidationReport(classifier, X, y, pos_label="A")
    display = getattr(report.metrics, metric)()
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()


def test_seed_none(linear_regression_data):
    """If `seed` is None (the default) the call should not be cached."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.prediction_error(seed=None)
    # skore should store the y_pred of the internal estimators, but not the plot
    assert report._cache == {}
