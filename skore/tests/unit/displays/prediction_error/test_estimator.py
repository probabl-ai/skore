import matplotlib as mpl
import pytest

from skore import EstimatorReport


def test_legend(
    pyplot,
    estimator_reports_regression_figure_axes,
):
    """Check the legend of the prediction error plot."""
    figure, _ = estimator_reports_regression_figure_axes
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 1
    assert legend_texts[0] == "Perfect predictions"


def test_legend_actual_vs_predicted(
    pyplot,
    estimator_report_regression_0,
):
    """Check the legend when kind is actual_vs_predicted."""
    report = estimator_report_regression_0
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 1
    assert legend_texts[0] == "Perfect predictions"


def test_invalid_subplot_by(estimator_report_regression_0):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = estimator_report_regression_0
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid `subplot_by` parameter. Valid options are: "
            "auto, None. Got 'invalid' instead."
        ),
    ):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize("subplot_by", [None, "auto"])
def test_valid_subplot_by(pyplot, estimator_report_regression_0, subplot_by):
    """Check that we can pass valid values to `subplot_by`."""
    report = estimator_report_regression_0
    display = report.metrics.prediction_error()
    display.plot(subplot_by=subplot_by)
    assert isinstance(display.ax_, mpl.axes.Axes)


def test_subplot_by_data_source(pyplot, estimator_report_regression_0):
    """Check the behaviour when `subplot_by` is `data_source`."""
    report = estimator_report_regression_0
    display = report.metrics.prediction_error(data_source="both")
    display.plot(subplot_by="data_source")
    assert isinstance(display.ax_[0], mpl.axes.Axes)
    assert len(display.ax_) == 2
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 1
    assert legend_texts[0] == "Perfect predictions"


def test_source_both(pyplot, estimator_report_regression_0):
    """Check the behaviour of the plot when data_source='both'."""
    report = estimator_report_regression_0
    display = report.metrics.prediction_error(data_source="both")
    display.plot()
    assert len(display.figure_.legends) == 1
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert legend_texts[-1] == "Perfect predictions"
    assert "train" in legend_texts
    assert "test" in legend_texts


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"subsample": -1}, "When an integer, subsample=-1 should be"),
        ({"subsample": 20.0}, "When a floating-point, subsample=20.0 should be"),
        ({"subsample": -20.0}, "When a floating-point, subsample=-20.0 should be"),
    ],
)
def test_wrong_subsample(pyplot, params, err_msg, estimator_report_regression_0):
    """Check that we raise the proper error when making the parameters validation."""
    report = estimator_report_regression_0
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.prediction_error(**params)


def test_pass_kind_to_plot(pyplot, estimator_report_regression_0):
    """Check that we raise an error when passing an invalid `kind` to plot."""
    report = estimator_report_regression_0
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "`kind` must be one of actual_vs_predicted, residual_vs_predicted. Got "
            "'invalid' instead."
        ),
    ):
        display.plot(kind="invalid")


def test_random_state(linear_regression_with_train_test):
    """If random_state is None (the default) the call should not be cached."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report.metrics.prediction_error()
    assert len(report._cache) == 2
