import matplotlib as mpl
import pytest

mpl.rc("figure", max_open_warning=False)


@pytest.fixture
def estimator_reports_regression_figure_axes(pyplot, estimator_report_regression_0):
    report = estimator_report_regression_0
    display = report.metrics.prediction_error()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture
def cross_validation_reports_regression_figure_axes(
    pyplot, cross_validation_report_regression_0
):
    report = cross_validation_report_regression_0
    display = report.metrics.prediction_error()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture
def comparison_estimator_reports_regression_figure_axes(
    pyplot, comparison_estimator_reports_regression
):
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture
def comparison_cross_validation_reports_regression_figure_axes(
    pyplot, comparison_cross_validation_reports_regression
):
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    display.plot()
    return display.figure_, display.ax_
