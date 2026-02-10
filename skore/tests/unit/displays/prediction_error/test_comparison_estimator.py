import matplotlib as mpl
import pytest


def test_legend(
    pyplot,
    comparison_estimator_reports_regression_figure_axes,
):
    """Check the legend of the prediction error plot with comparison of estimators."""
    figure, _ = comparison_estimator_reports_regression_figure_axes
    assert len(figure.legends) == 1
    legend_texts = [t.get_text() for t in figure.legends[0].get_texts()]
    assert len(legend_texts) == 1
    assert legend_texts[0] == "Perfect predictions"


def test_legend_actual_vs_predicted(pyplot, comparison_estimator_reports_regression):
    """Check the legend when kind is actual_vs_predicted."""
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 1
    assert legend_texts[0] == "Perfect predictions"


def test_invalid_subplot_by(pyplot, comparison_estimator_reports_regression):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid `subplot_by` parameter. Valid options are: "
            "auto, estimator. Got 'invalid' instead."
        ),
    ):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize("subplot_by", ["auto", "estimator"])
def test_valid_subplot_by(pyplot, comparison_estimator_reports_regression, subplot_by):
    """Check that we can pass valid values to `subplot_by`."""
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error()
    display.plot(subplot_by=subplot_by)
    assert isinstance(display.ax_[0], mpl.axes.Axes)
    assert len(display.ax_) == len(report.reports_)


def test_subplot_by_data_source(pyplot, comparison_estimator_reports_regression):
    """Check the behaviour when `subplot_by` is `data_source`."""
    report = comparison_estimator_reports_regression
    display = report.metrics.prediction_error(data_source="both")
    display.plot(subplot_by="data_source")
    assert len(display.ax_) == 2
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == "DummyRegressor_1"
    assert legend_texts[1] == "DummyRegressor_2"
    assert legend_texts[-1] == "Perfect predictions"


def test_source_both(pyplot, linear_regression_comparison_report):
    """Check the behaviour of the plot when data_source='both'."""
    report = linear_regression_comparison_report
    display = report.metrics.prediction_error(data_source="both")
    display.plot()
    assert len(display.figure_.legends) == 1
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == "train"
    assert legend_texts[1] == "test"
    assert legend_texts[2] == "Perfect predictions"
