import pytest


@pytest.mark.parametrize(
    "task, n_legend_entries",
    [("regression", 1), ("multioutput_regression", 3)],
)
def test_legend(pyplot, task, n_legend_entries, request):
    """Check the legend of the prediction error plot with comparison of estimators."""
    figure, _ = request.getfixturevalue(
        f"comparison_estimator_reports_{task}_figure_axes"
    )
    legend = figure.axes[len(figure.axes)//2].get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == n_legend_entries
    if task == "multioutput_regression":
        assert legend_texts[0] == "Output #0"
        assert legend_texts[1] == "Output #1"
    assert legend_texts[-1] == "Perfect predictions"


@pytest.mark.parametrize(
    "task, n_legend_entries",
    [("regression", 1), ("multioutput_regression", 3)],
)
def test_legend_actual_vs_predicted(pyplot, task, n_legend_entries, request):
    """Check the legend when kind is actual_vs_predicted."""
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in display.figure_.axes[len(display.figure_.axes)//2].get_legend().get_texts()]
    assert len(legend_texts) == n_legend_entries
    if task == "multioutput_regression":
        assert legend_texts[0] == "Output #0"
        assert legend_texts[1] == "Output #1"
    assert legend_texts[-1] == "Perfect predictions"


@pytest.mark.parametrize(
    "task, valid_values",
    [
        ("regression", ["auto", "estimator"]),
        ("multioutput_regression", ["auto", "output", "estimator"]),
    ],
)
def test_invalid_subplot_by(pyplot, task, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
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
        ("comparison_estimator_reports_regression", [("estimator", 2)]),
        (
            "comparison_estimator_reports_multioutput_regression",
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


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_subplot_by_data_source(pyplot, task, request):
    """Check the behaviour when `subplot_by` is `data_source`."""
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.metrics.prediction_error(data_source="both")
    if task == "multioutput_regression":
        with pytest.raises(
            ValueError,
            match="Invalid `subplot_by` parameter."
            + " Valid options are: auto, output, estimator. Got 'data_source' instead.",
        ):
            display.plot(subplot_by="data_source")
    else:
        display.plot(subplot_by="data_source")
        assert len(display.ax_) == 2
        legend_texts = [t.get_text() for t in display.figure_.axes[len(display.figure_.axes)//2].get_legend().get_texts()]
        assert len(legend_texts) == 3
        assert legend_texts[0] == "DummyRegressor_1"
        assert legend_texts[1] == "DummyRegressor_2"
        assert legend_texts[-1] == "Perfect predictions"


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_source_both(pyplot, task, request):
    """Check the behaviour of the plot when data_source='both'."""
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.metrics.prediction_error(data_source="both")
    display.plot()
    legend = display.figure_.axes[len(display.figure_.axes)//2].get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert len(legend_texts) == 3 if task == "regression" else 7
    assert legend_texts[-1] == "Perfect predictions"
    assert "train" in legend_texts
    assert "test" in legend_texts
    if task == "multioutput_regression":
        assert "0" in legend_texts
        assert "1" in legend_texts
