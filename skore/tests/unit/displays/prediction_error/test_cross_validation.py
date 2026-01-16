import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from skore import CrossValidationReport
from skore._sklearn._plot import PredictionErrorDisplay
from skore._sklearn._plot.metrics.prediction_error import RangeData
from skore._utils._testing import check_frame_structure


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_regression(pyplot, linear_regression_data, data_source):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with cross-validation data."""
    (estimator, X, y), cv = linear_regression_data, 3
    if data_source == "X_y":
        prediction_error_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        prediction_error_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error(**prediction_error_kwargs)
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert display._prediction_error["split"].nunique() == cv
    assert display.data_source == data_source
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
    assert len(display.lines_) == 1
    assert isinstance(display.lines_[0], mpl.lines.Line2D)
    assert display.lines_[0].get_color() == "black"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts
    # For cross-validation, we should have split labels
    assert any("Split" in text for text in legend_texts)

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"


def test_regression_actual_vs_predicted(pyplot, linear_regression_data):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert display._prediction_error["split"].nunique() == cv
    assert display.data_source == "test"

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    assert isinstance(display.lines_[0], mpl.lines.Line2D)
    assert display.lines_[0].get_color() == "black"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts
    # For cross-validation, we should have split labels
    assert any("Split" in text for text in legend_texts)

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"


def test_kwargs(pyplot, linear_regression_data):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a cross-validation report."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()

    display.set_style(
        relplot_kwargs={"palette": ["red", "green", "blue"]},
        perfect_model_kwargs={"color": "orange"},
    ).plot()
    assert display.lines_[0].get_color() == "orange"
    rgb_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for idx, collection in enumerate(display.ax_.collections):
        np.testing.assert_array_equal(
            collection.get_facecolor()[0][:3], rgb_colors[idx]
        )


def test_frame(linear_regression_data):
    """Test the frame method with cross-validation data."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    df = display.frame()

    expected_index = ["split"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv


def test_legend(pyplot, linear_regression_data):
    """Check the rendering of the legend for prediction error with an
    `CrossValidationReport`."""

    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    display.plot()
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts

    display.plot(kind="actual_vs_predicted")
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts

    cv = 10
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    display.plot()
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts

    display.plot(kind="actual_vs_predicted")
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts


def test_constructor(linear_regression_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()

    index_columns = ["estimator", "split"]
    df = display._prediction_error
    assert all(col in df.columns for col in index_columns)
    assert df["estimator"].unique() == report.estimator_name_
    assert df["split"].unique().tolist() == list(range(cv))


@pytest.mark.parametrize("subplot_by", [None, "split", "auto", "invalid"])
def test_subplot_by(pyplot, linear_regression_data, subplot_by):
    """Check that the subplot_by parameter works correctly for cross-val reports."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    if subplot_by == "invalid":
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: auto, split, None. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    elif subplot_by == "split":
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_[0], mpl.axes.Axes)
        assert len(display.ax_) == cv
    else:
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_, mpl.axes.Axes)


def test_title(pyplot, linear_regression_data):
    """Check that the title contains expected elements."""
    (estimator, X, y), cv = linear_regression_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.prediction_error()
    display.plot()
    title = display.figure_._suptitle.get_text()
    assert "Prediction Error" in title
    assert report.estimator_name_ in title
    assert "Data source: Test set" in title
