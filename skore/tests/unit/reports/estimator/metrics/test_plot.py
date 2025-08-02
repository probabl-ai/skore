import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport, RocCurveDisplay


def test_estimator_report_plot_roc(forest_binary_classification_with_test):
    """Check that the ROC plot method works."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert isinstance(report.metrics.roc(), RocCurveDisplay)


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification(
    pyplot, forest_binary_classification_with_test, display
):
    """The call to display functions should be cached."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)()
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)()
    assert display_first_call is display_second_call


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = getattr(report.metrics, metric)()
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression(
    pyplot, linear_regression_with_test, display
):
    """The call to display functions should be cached, as long as the arguments make it
    reproducible."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(seed=0)
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(seed=0)
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification_external_data(
    pyplot, forest_binary_classification_with_test, display
):
    """The call to display functions should be cached when passing external data."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression_external_data(
    pyplot, linear_regression_with_test, display
):
    """The call to display functions should be cached when passing external data,
    as long as the arguments make it reproducible."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test, seed=0
    )
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test, seed=0
    )
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification_switching_data_source(
    pyplot, forest_binary_classification_with_test, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call
    display_third_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is not display_third_call
    assert display_second_call is not display_third_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression_switching_data_source(
    pyplot, linear_regression_with_test, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call
    display_third_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is not display_third_call
    assert display_second_call is not display_third_call
