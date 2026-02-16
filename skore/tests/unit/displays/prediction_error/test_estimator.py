import matplotlib as mpl
import pytest

from skore import EstimatorReport


@pytest.mark.parametrize(
    "task, n_legend_entries",
    [("regression", 1), ("multioutput_regression", 3)],
)
def test_legend(pyplot, task, n_legend_entries, request):
    """Check the legend of the prediction error plot."""
    figure, _ = request.getfixturevalue(f"estimator_reports_{task}_figure_axes")
    assert len(figure.legends) == 1
    legend = figure.legends[0]
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
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == n_legend_entries
    if task == "multioutput_regression":
        assert legend_texts[0] == "Output #0"
        assert legend_texts[1] == "Output #1"
    assert legend_texts[-1] == "Perfect predictions"


@pytest.mark.parametrize(
    "task, valid_values",
    [
        ("regression", ["auto", "None"]),
        ("multioutput_regression", ["auto", "output", "None"]),
    ],
)
def test_invalid_subplot_by(pyplot, task, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
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
        ("estimator_reports_regression", [(None, 0)]),
        (
            "estimator_reports_multioutput_regression",
            [("output", 2), (None, 0)],
        ),
    ],
)
def test_valid_subplot_by(pyplot, fixture_name, subplot_by_tuples, request):
    """Check that we can pass valid values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.prediction_error()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_subplot_by_data_source(pyplot, task, request):
    """Check the behaviour when `subplot_by` is `data_source`."""
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.metrics.prediction_error(data_source="both")
    display.plot(subplot_by="data_source")
    assert isinstance(display.ax_[0], mpl.axes.Axes)
    assert len(display.ax_) == 2
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 1 if task == "regression" else 2
    assert legend_texts[-1] == "Perfect predictions"
    if task == "multioutput_regression":
        assert legend_texts[0] == "Output #0"
        assert legend_texts[1] == "Output #1"


@pytest.mark.parametrize("task", ["regression", "multioutput_regression"])
def test_source_both(pyplot, task, request):
    """Check the behaviour of the plot when data_source='both'."""
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.metrics.prediction_error(data_source="both")
    display.plot()
    assert len(display.figure_.legends) == 1
    legend_texts = [t.get_text() for t in display.figure_.legends[0].get_texts()]
    assert legend_texts[-1] == "Perfect predictions"
    assert "train" in legend_texts
    assert "test" in legend_texts
    if task == "multioutput_regression":
        assert "0" in legend_texts
        assert "1" in legend_texts


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"subsample": -1}, "When an integer, subsample=-1 should be"),
        ({"subsample": 20.0}, "When a floating-point, subsample=20.0 should be"),
        ({"subsample": -20.0}, "When a floating-point, subsample=-20.0 should be"),
    ],
)
def test_wrong_subsample(pyplot, params, err_msg, estimator_reports_regression):
    """Check that we raise the proper error when making the parameters validation."""
    report = estimator_reports_regression[0]
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.prediction_error(**params)


def test_pass_kind_to_plot(pyplot, estimator_reports_regression):
    """Check that we raise an error when passing an invalid `kind` to plot."""
    report = estimator_reports_regression[0]
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
