import matplotlib as mpl
import pytest
from sklearn.metrics import make_scorer, precision_score, r2_score

mpl.rc("figure", max_open_warning=False)


@pytest.fixture(scope="module")
def estimator_type():
    return "linear"


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    pyplot, estimator_reports_binary_classification
):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multiclass_classification_figure_axes(
    pyplot, estimator_reports_multiclass_classification
):
    report = estimator_reports_multiclass_classification[0]
    metric = make_scorer(precision_score, average=None)
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric="precision score")
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_regression_figure_axes(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multioutput_regression_figure_axes(
    pyplot, estimator_reports_multioutput_regression
):
    report = estimator_reports_multioutput_regression[0]
    metric = make_scorer(r2_score, multioutput="raw_values")
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric="r2 score")
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    pyplot, cross_validation_reports_binary_classification
):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification_figure_axes(
    pyplot, cross_validation_reports_multiclass_classification
):
    report = cross_validation_reports_multiclass_classification[0]
    metric = make_scorer(precision_score, average=None)
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric="precision score")
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_regression_figure_axes(
    pyplot, cross_validation_reports_regression
):
    report = cross_validation_reports_regression[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multioutput_regression_figure_axes(
    pyplot, cross_validation_reports_multioutput_regression
):
    report = cross_validation_reports_multioutput_regression[0]
    metric = make_scorer(r2_score, multioutput="raw_values")
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric="r2 score")
    return display.figure_, display.ax_
