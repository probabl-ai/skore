import pytest


@pytest.fixture
def estimator_reports_regression_figure_axes(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def cross_validation_reports_regression_figure_axes(
    cross_validation_reports_regression,
):
    report = cross_validation_reports_regression[0]
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def comparison_estimator_reports_regression_figure_axes(
    comparison_estimator_reports_regression,
):
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def comparison_cross_validation_reports_regression_figure_axes(
    comparison_cross_validation_reports_regression,
):
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def estimator_reports_multioutput_regression_figure_axes(
    estimator_reports_multioutput_regression,
):
    report = estimator_reports_multioutput_regression[0]
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def cross_validation_reports_multioutput_regression_figure_axes(
    cross_validation_reports_multioutput_regression,
):
    report = cross_validation_reports_multioutput_regression[0]
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def comparison_estimator_reports_multioutput_regression_figure_axes(
    comparison_estimator_reports_multioutput_regression,
):
    report = comparison_estimator_reports_multioutput_regression
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture
def comparison_cross_validation_reports_multioutput_regression_figure_axes(
    comparison_cross_validation_reports_multioutput_regression,
):
    report = comparison_cross_validation_reports_multioutput_regression
    display = report.metrics.prediction_error()
    fig = display.plot()

    return fig, fig.axes
