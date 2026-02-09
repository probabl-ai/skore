import matplotlib as mpl
import numpy as np
import pytest


def test_legend(
    pyplot,
    comparison_cross_validation_reports_regression_figure_axes,
):
    """Check the legend of the prediction error plot with comparison crossvalidation."""
    figure, _ = comparison_cross_validation_reports_regression_figure_axes
    assert len(figure.legends) == 1
    legend_texts = [t.get_text() for t in figure.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == "Split #0"
    assert legend_texts[1] == "Split #1"
    assert legend_texts[2] == "Perfect predictions"


def test_legend_actual_vs_predicted(
    pyplot, comparison_cross_validation_reports_regression
):
    """Check the legend when kind is actual_vs_predicted."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    facet = display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in facet.figure.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == "Split #0"
    assert legend_texts[1] == "Split #1"
    assert legend_texts[2] == "Perfect predictions"

    for ax in facet.axes.flatten():
        assert ax.get_xlim() == ax.get_ylim()
        assert np.array_equal(ax.get_xticks(), ax.get_yticks())


def test_invalid_subplot_by(comparison_cross_validation_reports_regression):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid `subplot_by` parameter. Valid options are: "
            "auto, split, estimator. Got 'invalid' instead."
        ),
    ):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize("subplot_by", ["auto", "estimator", "split"])
def test_valid_subplot_by(
    pyplot, comparison_cross_validation_reports_regression, subplot_by
):
    """Check that we can pass valid values to `subplot_by`."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    facet = display.plot(subplot_by=subplot_by)
    ax_ = facet.axes.flatten()
    assert isinstance(ax_[0], mpl.axes.Axes)
    if subplot_by == "estimator":
        assert len(ax_) == len(report.reports_)
    elif subplot_by == "split":
        n_splits = len(next(iter(report.reports_.values())).estimator_reports_)
        assert len(ax_) == n_splits
