import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from skore import ComparisonReport, EstimatorReport
from skore.sklearn._plot import PredictionErrorDisplay
from skore.sklearn._plot.metrics.prediction_error import RangeData
from skore.utils._testing import check_frame_structure, check_legend_position


def test_regression(pyplot, regression_data):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.prediction_error, pd.DataFrame)
    assert list(display.prediction_error["estimator_name"].unique()) == [
        "estimator_1",
        "estimator_2",
    ]
    assert display.data_source == "test"
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
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Residuals (actual - predicted)"

    assert display.ax_.get_aspect() not in ("equal", 1.0)


def test_regression_actual_vs_predicted(pyplot, regression_data):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()
    display.plot(kind="actual_vs_predicted")
    assert isinstance(display, PredictionErrorDisplay)

    # check the structure of the attributes
    assert isinstance(display.prediction_error, pd.DataFrame)
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
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Predicted values"
    assert display.ax_.get_ylabel() == "Actual values"

    assert display.ax_.get_aspect() in ("equal", 1.0)


def test_kwargs(pyplot, regression_data):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
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
def test_wrong_kwargs(pyplot, regression_data, data_points_kwargs):
    """Check that we raise an error when we pass keyword arguments to the prediction
    error plot if there is a comparison report."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        },
    )
    display = report.metrics.prediction_error()

    err_msg = (
        "You intend to plot prediction errors either from multiple estimators "
        "or from a cross-validated estimator. We expect `data_points_kwargs` to be "
        "a list of dictionaries with the same length as the number of "
        "estimators or splits."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(data_points_kwargs=data_points_kwargs)


def test_frame(regression_data):
    """Test the frame method with comparison data."""
    estimator, X_train, X_test, y_train, y_test = regression_data
    estimator_2 = clone(estimator).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.prediction_error()
    df = display.frame()

    expected_index = ["estimator_name"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == 2


def test_legend(pyplot, regression_data):
    """Check the rendering of the legend for prediction error with a
    `ComparisonReport`."""

    estimator, X_train, X_test, y_train, y_test = regression_data
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator 1": report_1, "estimator 2": report_2}
    )
    display = report.metrics.prediction_error()
    display.plot()
    # The loc doesn't matter because bbox_to_anchor is used
    check_legend_position(display.ax_, loc="upper left", position="outside")

    display.plot(kind="actual_vs_predicted")
    check_legend_position(display.ax_, loc="lower right", position="inside")

    reports = {
        f"estimator {i}": EstimatorReport(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        for i in range(1, 10)
    }
    report = ComparisonReport(reports=reports)
    display = report.metrics.prediction_error()
    display.plot()
    # The loc doesn't matter because bbox_to_anchor is used
    check_legend_position(display.ax_, loc="upper left", position="outside")

    display.plot(kind="actual_vs_predicted")
    check_legend_position(display.ax_, loc="upper left", position="outside")
