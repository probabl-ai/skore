import matplotlib as mpl
import numpy as np
import pandas as pd

from skore import ComparisonReport, CrossValidationReport
from skore._sklearn._plot import PredictionErrorDisplay
from skore._sklearn._plot.metrics.prediction_error import RangeData
from skore._utils._testing import check_frame_structure


def test_regression(pyplot, comparison_cross_validation_reports_regression):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with a comparison report."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert list(display._prediction_error["estimator"].unique()) == [
        "DummyRegressor_1",
        "DummyRegressor_2",
    ]
    assert display.data_source == "test"
    assert isinstance(display.range_y_true, RangeData)
    assert isinstance(display.range_y_pred, RangeData)
    assert isinstance(display.range_residuals, RangeData)
    for attr in ("y_true", "y_pred", "residuals"):
        global_min, global_max = np.inf, -np.inf
        for display_attr in display._prediction_error[attr]:
            global_min = min(global_min, np.min(display_attr))
            global_max = max(global_max, np.max(display_attr))
        assert getattr(display, f"range_{attr}").min == global_min
        assert getattr(display, f"range_{attr}").max == global_max

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_color() == "black"

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == 2
    for ax in display.ax_:
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "Predicted values"
        assert ax.get_ylabel() == "Residuals (actual - predicted)"

    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts


def test_regression_actual_vs_predicted(
    pyplot, comparison_cross_validation_reports_regression
):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert display.data_source == "test"

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 2
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_color() == "black"

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == 2
    for ax in display.ax_:
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "Predicted values"
        assert ax.get_ylabel() == "Actual values"

    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts


def test_kwargs(pyplot, comparison_cross_validation_reports_regression):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a comparison report."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()

    display.set_style(
        relplot_kwargs={"palette": ["red", "blue", "green", "magenta", "cyan"]},
        perfect_model_kwargs={"color": "orange"},
    ).plot()
    for line in display.lines_:
        assert line.get_color() == "orange"
    rgb_colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    for ax in display.ax_:
        assert len(ax.collections) > 0
        for idx, collection in enumerate(ax.collections):
            np.testing.assert_array_equal(
                collection.get_facecolor()[0][:3], rgb_colors[idx]
            )


def test_constructor(linear_regression_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = linear_regression_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    # add a different number of splits for the second report
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.prediction_error()

    index_columns = ["estimator", "split"]
    df = display._prediction_error
    assert all(col in df.columns for col in index_columns)
    assert df.query("estimator == 'estimator_1'")["split"].unique().tolist() == list(
        range(cv)
    )
    assert df.query("estimator == 'estimator_2'")["split"].unique().tolist() == list(
        range(cv + 1)
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())


def test_frame(comparison_cross_validation_reports_regression):
    """Test the frame method with regression comparison cross-validation data."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    df = display.frame()

    expected_index = ["estimator", "split"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == len(report.reports_)
