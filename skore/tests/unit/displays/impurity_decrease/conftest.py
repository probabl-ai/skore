import matplotlib as mpl
import pytest

mpl.rc("figure", max_open_warning=False)


@pytest.fixture(scope="module")
def estimator_type():
    return "forest"


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    pyplot, estimator_reports_binary_classification
):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multiclass_classification_figure_axes(
    pyplot, estimator_reports_multiclass_classification
):
    report = estimator_reports_multiclass_classification[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_regression_figure_axes(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multioutput_regression_figure_axes(
    pyplot, estimator_reports_multioutput_regression
):
    report = estimator_reports_multioutput_regression[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    pyplot, cross_validation_reports_binary_classification
):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification_figure_axes(
    pyplot, cross_validation_reports_multiclass_classification
):
    report = cross_validation_reports_multiclass_classification[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_regression_figure_axes(
    pyplot, cross_validation_reports_regression
):
    report = cross_validation_reports_regression[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multioutput_regression_figure_axes(
    pyplot, cross_validation_reports_multioutput_regression
):
    report = cross_validation_reports_multioutput_regression[0]
    display = report.inspection.impurity_decrease()
    display.plot()
    return display.figure_, display.ax_
