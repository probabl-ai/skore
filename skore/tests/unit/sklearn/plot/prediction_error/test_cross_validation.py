import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from skore import CrossValidationReport
from skore._sklearn._plot import PredictionErrorDisplay
from skore._sklearn._plot.metrics.prediction_error import RangeData
from skore.utils._testing import check_frame_structure, check_legend_position


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_regression(pyplot, regression_data_no_split, data_source):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with cross-validation data."""
    (estimator, X, y), cv = regression_data_no_split, 3
    if data_source == "X_y":
        prediction_error_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        prediction_error_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error(**prediction_error_kwargs)
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.prediction_error, pd.DataFrame)
    assert display.prediction_error["split_index"].nunique() == cv
    assert display.data_source == data_source
    assert isinstance(display.range_y_true, RangeData)
    assert isinstance(display.range_y_pred, RangeData)
    assert isinstance(display.range_residuals, RangeData)
    for attr in ("y_true", "y_pred", "residuals"):
        global_min, global_max = np.inf, -np.inf
        for display_attr in display.prediction_error[attr]:
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
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert legend.get_title().get_text() == f"{data_source_title.capitalize()} set"
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


def test_regression_actual_vs_predicted(pyplot, regression_data_no_split):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.prediction_error, pd.DataFrame)
    assert display.prediction_error["split_index"].nunique() == cv
    assert display.data_source == "test"

    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_label() == "Perfect predictions"
    assert display.line_.get_color() == "black"

    assert isinstance(display.scatter_, list)
    for scatter in display.scatter_:
        assert isinstance(scatter, mpl.collections.PathCollection)

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_kwargs(pyplot, regression_data_no_split):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a cross-validation report."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot(
        data_points_kwargs=[{"color": "red"}, {"color": "green"}, {"color": "blue"}],
        perfect_model_kwargs={"color": "orange"},
    )
    rgb_colors = [
        [[1.0, 0.0, 0.0, 0.3]],
        [[0.0, 0.50196078, 0.0, 0.3]],
        [[0.0, 0.0, 1.0, 0.3]],
    ]
    for scatter, rgb_color in zip(display.scatter_, rgb_colors, strict=False):
        np.testing.assert_allclose(scatter.get_facecolor(), rgb_color, rtol=1e-3)
    assert display.line_.get_color() == "orange"


@pytest.mark.parametrize("data_points_kwargs", ["not a list", [{"color": "red"}]])
def test_wrong_kwargs(pyplot, regression_data_no_split, data_points_kwargs):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a cross-validation report."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot prediction errors either from multiple estimators "
        "or from a cross-validated estimator. We expect `data_points_kwargs` to be "
        "a list of dictionaries with the same length as the number of "
        "estimators or splits."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


def test_frame(regression_data_no_split):
    """Test the frame method with cross-validation data."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    df = display.frame()

    expected_index = ["split_index"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split_index"].nunique() == cv


def test_legend(pyplot, regression_data_no_split):
    """Check the rendering of the legend for prediction error with an
    `CrossValidationReport`."""

    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")

    display.plot(kind="actual_vs_predicted")
    check_legend_position(display.ax_, loc="lower right", position="inside")

    cv = 10
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")

    display.plot(kind="actual_vs_predicted")
    check_legend_position(display.ax_, loc="upper left", position="outside")


def test_constructor(regression_data_no_split):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = regression_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.prediction_error()

    index_columns = ["estimator_name", "split_index"]
    for df in [display.prediction_error]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split_index"].unique().tolist() == list(range(cv))
