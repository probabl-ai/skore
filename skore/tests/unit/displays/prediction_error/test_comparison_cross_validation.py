import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from skore import ComparisonReport, CrossValidationReport
from skore._sklearn._plot import PredictionErrorDisplay
from skore._sklearn._plot.metrics.prediction_error import RangeData
from skore._utils._testing import check_frame_structure, check_legend_position


def test_regression(pyplot, comparison_cross_validation_reports_regression):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with a comparison report."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert list(display._prediction_error["estimator_name"].unique()) == [
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
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    # The loc doesn't matter because bbox_to_anchor is used
    check_legend_position(display.ax_, loc="upper left", position="outside")
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


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

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    check_legend_position(display.ax_, loc="lower right", position="inside")
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_kwargs(pyplot, comparison_cross_validation_reports_regression):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a comparison report."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    display.plot(
        data_points_kwargs=[{"color": "red"}, {"color": "blue"}],
        perfect_model_kwargs={"color": "orange"},
    )
    rgb_colors = [[[1.0, 0.0, 0.0, 0.3]], [[0.0, 0.0, 1.0, 0.3]]]
    for scatter, rgb_color in zip(display.scatter_, rgb_colors, strict=False):
        np.testing.assert_allclose(scatter.get_facecolor(), rgb_color, rtol=1e-3)
    assert display.line_.get_color() == "orange"


@pytest.mark.parametrize("data_points_kwargs", ["not a list", [{"color": "red"}]])
def test_wrong_kwargs(
    pyplot, comparison_cross_validation_reports_regression, data_points_kwargs
):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a comparison report."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot prediction errors either from multiple estimators "
        "or from a cross-validated estimator. We expect `data_points_kwargs` to be "
        "a list of dictionaries with the same length as the number of "
        "estimators or splits."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


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

    index_columns = ["estimator_name", "split"]
    df = display._prediction_error
    assert all(col in df.columns for col in index_columns)
    assert df.query("estimator_name == 'estimator_1'")[
        "split"
    ].unique().tolist() == list(range(cv))
    assert df.query("estimator_name == 'estimator_2'")[
        "split"
    ].unique().tolist() == list(range(cv + 1))
    assert df["estimator_name"].unique().tolist() == list(report.reports_.keys())


def test_frame(comparison_cross_validation_reports_regression):
    """Test the frame method with regression comparison cross-validation data."""
    report = comparison_cross_validation_reports_regression
    display = report.metrics.prediction_error()
    df = display.frame()

    expected_index = ["estimator_name", "split"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == len(report.reports_)
