import matplotlib as mpl
import pytest

mpl.rc("figure", max_open_warning=False)


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    pyplot, estimator_report_binary_classification_0
):
    report = estimator_report_binary_classification_0
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multiclass_classification_figure_axes(
    pyplot, estimator_report_multiclass_classification_0
):
    report = estimator_report_multiclass_classification_0
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    pyplot, cross_validation_report_binary_classification_0
):
    report = cross_validation_report_binary_classification_0
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification_figure_axes(
    pyplot, cross_validation_report_multiclass_classification_0
):
    report = cross_validation_report_multiclass_classification_0
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification_figure_axes(
    pyplot, comparison_estimator_reports_binary_classification
):
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification_figure_axes(
    pyplot, comparison_estimator_reports_multiclass_classification
):
    report = comparison_estimator_reports_multiclass_classification
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification_figure_axes(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification_figure_axes(
    pyplot, comparison_cross_validation_reports_multiclass_classification
):
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.confusion_matrix()
    display.plot()
    return display.figure_, display.ax_
