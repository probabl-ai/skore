import matplotlib as mpl
import pytest

mpl.rc("figure", max_open_warning=False)


@pytest.fixture
def estimator_reports_regression_figure_axes(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot()
    return facet.figure, facet.axes.flatten()


@pytest.fixture
def cross_validation_reports_regression_figure_axes(
    pyplot, cross_validation_reports_regression
):
    report = cross_validation_reports_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot()
    return facet.figure, facet.axes.flatten()


@pytest.fixture
def comparison_estimator_reports_regression_figure_axes(
    pyplot, comparison_estimator_reports_regression
):
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    facet = display.plot()
    return facet.figure, facet.axes.flatten()


@pytest.fixture
def comparison_cross_validation_reports_regression_figure_axes(
    pyplot, comparison_cross_validation_reports_regression
):
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    facet = display.plot()
    return facet.figure, facet.axes.flatten()


@pytest.fixture
def estimator_reports_multioutput_regression_figure_axes(
    pyplot, estimator_reports_multioutput_regression
):
    report = estimator_reports_multioutput_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot()
    return display.figure_, display.ax_.flatten()


@pytest.fixture
def cross_validation_reports_multioutput_regression_figure_axes(
    pyplot, cross_validation_reports_multioutput_regression
):
    report = cross_validation_reports_multioutput_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot()
    return display.figure_, display.ax_.flatten()


@pytest.fixture
def comparison_estimator_reports_multioutput_regression_figure_axes(
    pyplot, comparison_estimator_reports_multioutput_regression
):
    report = comparison_estimator_reports_multioutput_regression
    display = report.metrics.prediction_error()
    facet = display.plot()
    return display.figure_, display.ax_.flatten()


@pytest.fixture
def comparison_cross_validation_reports_multioutput_regression_figure_axes(
    pyplot, comparison_cross_validation_reports_multioutput_regression
):
    report = comparison_cross_validation_reports_multioutput_regression
    display = report.metrics.prediction_error()
    facet = display.plot()
    return display.figure_, display.ax_.flatten()

