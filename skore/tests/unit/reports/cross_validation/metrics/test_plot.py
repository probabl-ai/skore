import numpy as np
import pytest

from skore import CrossValidationReport, RocCurveDisplay
from skore._utils._testing import check_cache_unchanged


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
    _ = getattr(report.metrics, display)()
    child_cache = report.estimator_reports_[0]._cache
    assert child_cache != {}
    with check_cache_unchanged(child_cache):
        _ = getattr(report.metrics, display)()


@pytest.mark.parametrize("display", ["prediction_error"])
def test_display_regression(pyplot, linear_regression_data, display):
    """General behaviour of the function creating display on regression."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.metrics, display)
    _ = getattr(report.metrics, display)(seed=0)
    child_cache = report.estimator_reports_[0]._cache
    assert child_cache != {}
    with check_cache_unchanged(child_cache):
        _ = getattr(report.metrics, display)(seed=0)


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_pos_label(
    pyplot, metric, forest_binary_classification_data
):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    classifier, X, y = forest_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report = CrossValidationReport(classifier, X, y)
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label" not in fig.get_suptitle()
    fig = display.plot(label="A")
    assert "Positive label: A" in fig.get_suptitle()

    report = CrossValidationReport(classifier, X, y, pos_label="A")
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label: A" in fig.get_suptitle()

    report = CrossValidationReport(classifier, X, y, pos_label="B")
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label: B" in fig.get_suptitle()


def test_seed_none(linear_regression_data):
    """If `seed` is None (the default) the call should not be cached."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.prediction_error(seed=None)
    # skore stores the predictions of the internal estimators, but not the
    # concatenated cross-validation display.
    assert all(
        len(estimator_report._cache) == 2
        for estimator_report in report.estimator_reports_
    )
