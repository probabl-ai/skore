import numpy as np
import pytest


@pytest.mark.parametrize(
    "task, legend_prefix",
    [("regression", "Split"), ("multioutput_regression", "Output")],
)
def test_legend(task, legend_prefix, request):
    """Check the legend of the prediction error plot with comparison crossvalidation."""
    figure, _ = request.getfixturevalue(
        f"comparison_cross_validation_reports_{task}_figure_axes"
    )
    legend = figure.axes[len(figure.axes) // 2].get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == f"{legend_prefix} #0"
    assert legend_texts[1] == f"{legend_prefix} #1"
    assert legend_texts[2] == "Perfect predictions"


@pytest.mark.parametrize(
    "task, legend_prefix",
    [("regression", "Split"), ("multioutput_regression", "Output")],
)
def test_legend_actual_vs_predicted(task, legend_prefix, request):
    """Check the legend when kind is actual_vs_predicted."""
    report = request.getfixturevalue(f"comparison_cross_validation_reports_{task}")
    display = report.metrics.prediction_error()
    fig = display.plot(kind="actual_vs_predicted")
    axes = fig.axes
    legend = fig.axes[len(fig.axes) // 2].get_legend()
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 3
    assert legend_texts[0] == f"{legend_prefix} #0"
    assert legend_texts[1] == f"{legend_prefix} #1"
    assert legend_texts[2] == "Perfect predictions"

    for ax in axes:
        assert ax.get_xlim() == ax.get_ylim()
        assert np.array_equal(ax.get_xticks(), ax.get_yticks())


@pytest.mark.parametrize(
    "task, valid_values",
    [
        ("regression", ["auto", "split", "estimator"]),
        ("multioutput_regression", ["auto", "output", "estimator"]),
    ],
)
def test_invalid_subplot_by(task, valid_values, request):
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
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass valid values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.prediction_error()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert len(axes) == expected_len


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_subplot_by_data_source(task, request):
    """Check the behaviour when `subplot_by` is `data_source`."""
    report = request.getfixturevalue(f"comparison_cross_validation_reports_{task}")
    display = report.metrics.prediction_error(data_source="both")
    if task == "multioutput_regression":
        with pytest.raises(
            ValueError,
            match="Invalid `subplot_by` parameter."
            + " Valid options are: auto, output, estimator. Got 'data_source' instead.",
        ):
            display.plot(subplot_by="data_source")
    else:
        fig = display.plot(subplot_by="data_source")
        axes = fig.axes
        assert len(axes) == 2
        legend = fig.axes[len(fig.axes) // 2].get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert legend_texts == ["Split #0", "Split #1", "Perfect predictions"]


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_source_both(task, request):
    """Check the behaviour of the plot when `data_source='both'`."""
    report = request.getfixturevalue(f"comparison_cross_validation_reports_{task}")
    display = report.metrics.prediction_error(data_source="both")
    assert display.data_source == "both"
    plot_data = display.frame()
    assert "data_source" in plot_data.columns
    assert set(plot_data["data_source"]) == {"train", "test"}
    fig = display.plot()
    legend = fig.axes[len(fig.axes) // 2].get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert legend_texts[-1] == "Perfect predictions"
    assert "train" in legend_texts
    assert "test" in legend_texts
    if task == "regression":
        assert legend_texts == [
            "Split #0",
            "Split #1",
            "train",
            "test",
            "Perfect predictions",
        ]
    else:
        assert legend_texts == [
            "Output #0",
            "Output #1",
            "train",
            "test",
            "Perfect predictions",
        ]
