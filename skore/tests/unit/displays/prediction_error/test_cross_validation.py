import matplotlib as mpl
import pytest


def test_legend(
    pyplot,
    cross_validation_reports_regression_figure_axes,
):
    """Check the legend of the prediction error plot with cross-validation."""
    figure, _ = cross_validation_reports_regression_figure_axes
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == "Split #0"
    assert legend_texts[1] == "Split #1"
    assert legend_texts[2] == "Perfect predictions"


def test_legend_actual_vs_predicted(pyplot, cross_validation_reports_regression):
    """Check the legend when kind is actual_vs_predicted."""
    report = cross_validation_reports_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in facet.figure.legends[0].get_texts()]

    assert len(legend_texts) == 3
    assert legend_texts[0] == "Split #0"
    assert legend_texts[1] == "Split #1"
    assert legend_texts[2] == "Perfect predictions"


def test_invalid_subplot_by(pyplot, cross_validation_reports_regression):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = cross_validation_reports_regression[0]
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid `subplot_by` parameter. Valid options are: "
            "auto, split, None. Got 'invalid' instead."
        ),
    ):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize("subplot_by", [None, "split", "auto"])
def test_valid_subplot_by(pyplot, cross_validation_reports_regression, subplot_by):
    """Check that we can pass valid values to `subplot_by`."""
    report = cross_validation_reports_regression[0]
    display = report.metrics.prediction_error()
    facet = display.plot(subplot_by=subplot_by)
    if subplot_by == "split":
        ax_ = facet.axes.flatten()
        assert isinstance(ax_[0], mpl.axes.Axes)
        assert len(ax_) == len(report.estimator_reports_)
    else:
        ax = facet.axes.squeeze().item()
        assert isinstance(ax, mpl.axes.Axes)
