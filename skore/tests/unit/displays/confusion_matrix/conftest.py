import pytest


@pytest.fixture(scope="module")
def estimator_type():
    return "linear"


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    estimator_reports_binary_classification,
):
    report = estimator_reports_binary_classification[0]
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def estimator_reports_multiclass_classification_figure_axes(
    estimator_reports_multiclass_classification,
):
    report = estimator_reports_multiclass_classification[0]
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    cross_validation_reports_binary_classification,
):
    report = cross_validation_reports_binary_classification[0]
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification_figure_axes(
    cross_validation_reports_multiclass_classification,
):
    report = cross_validation_reports_multiclass_classification[0]
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification_figure_axes(
    comparison_estimator_reports_binary_classification,
):
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification_figure_axes(
    comparison_estimator_reports_multiclass_classification,
):
    report = comparison_estimator_reports_multiclass_classification
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification_figure_axes(
    comparison_cross_validation_reports_binary_classification,
):
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification_figure_axes(
    comparison_cross_validation_reports_multiclass_classification,
):
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.confusion_matrix()
    fig = display.plot()

    return fig, fig.axes
