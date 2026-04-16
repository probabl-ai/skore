import matplotlib as mpl
import pytest

mpl.rc("figure", max_open_warning=False)


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    pyplot, estimator_reports_binary_classification
):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.calibration_curve()
    fig = display.plot()
    return fig, fig.axes


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    pyplot, cross_validation_reports_binary_classification
):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.calibration_curve()
    fig = display.plot()
    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification_figure_axes(
    pyplot, comparison_estimator_reports_binary_classification
):
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.calibration_curve()
    fig = display.plot()
    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification_figure_axes(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    report = comparison_cross_validation_reports_binary_classification
    display = report.inspection.calibration_curve()
    fig = display.plot()
    return fig, fig.axes
