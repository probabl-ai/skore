import numpy as np
import pytest


@pytest.mark.parametrize(
    "task, legend_prefix",
    [("regression", "Split"), ("multioutput_regression", "Output")],
)
def test_legend(pyplot, task, legend_prefix, request):
    """Check the legend of the prediction error plot with comparison crossvalidation."""
    figure, _ = request.getfixturevalue(
        f"comparison_cross_validation_reports_{task}_figure_axes"
    )
    assert len(figure.legends) == 1
    legend_texts = [t.get_text() for t in figure.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == f"{legend_prefix} #0"
    assert legend_texts[1] == f"{legend_prefix} #1"
    assert legend_texts[2] == "Perfect predictions"


@pytest.mark.parametrize(
    "task, legend_prefix",
    [("regression", "Split"), ("multioutput_regression", "Output")],
)
def test_legend_actual_vs_predicted(pyplot, task, legend_prefix, request):
    """Check the legend when kind is actual_vs_predicted."""
    report = request.getfixturevalue(f"comparison_cross_validation_reports_{task}")
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == f"{legend_prefix} #0"
    assert legend_texts[1] == f"{legend_prefix} #1"
    assert legend_texts[2] == "Perfect predictions"

    for ax in display.ax_:
        assert ax.get_xlim() == ax.get_ylim()
        assert np.array_equal(ax.get_xticks(), ax.get_yticks())


@pytest.mark.parametrize(
    "task, valid_values",
    [
        ("regression", ["auto", "estimator", "split"]),
        ("multioutput_regression", ["auto", "output", "estimator"]),
    ],
)
def test_invalid_subplot_by(pyplot, task, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(f"comparison_cross_validation_reports_{task}")
    display = report.metrics.prediction_error()
    with pytest.raises(
        ValueError,
        match=(
            "Invalid `subplot_by` parameter. Valid options are: "
            f"{', '.join(valid_values)}. Got 'invalid' instead."
        ),
    ):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "comparison_cross_validation_reports_regression",
            [("estimator", 2), ("split", 2)],
        ),
        (
            "comparison_cross_validation_reports_multioutput_regression",
            [("output", 2), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(pyplot, fixture_name, subplot_by_tuples, request):
    """Check that we can pass valid values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.prediction_error()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        assert len(display.ax_) == expected_len
