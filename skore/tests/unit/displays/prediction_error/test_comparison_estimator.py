import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from skore import ComparisonReport, EstimatorReport
from skore._sklearn._plot import PredictionErrorDisplay
from skore._sklearn._plot.metrics.prediction_error import RangeData
from skore._utils._testing import check_frame_structure


def test_regression(pyplot, linear_regression_with_train_test):
    """Check the attributes and default plotting behaviour of the prediction error plot
    with a comparison report."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert list(display._prediction_error["estimator"].unique()) == [
        "estimator_1",
        "estimator_2",
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


def test_regression_actual_vs_predicted(pyplot, linear_regression_with_train_test):
    """Check the attributes when switching to the "actual_vs_predicted" kind."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    assert isinstance(display._prediction_error, pd.DataFrame)
    assert display.data_source == "test"

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 2  # One line per subplot (estimator)
    for line in display.lines_:
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_color() == "black"

    # For comparison reports, ax_ is an array of axes
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == 2
    for ax in display.ax_:
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_xlabel() == "Predicted values"
        assert ax.get_ylabel() == "Actual values"

    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts


def test_kwargs(pyplot, linear_regression_with_train_test):
    """Check that we can pass keyword arguments to the prediction error plot when
    there is a comparison report."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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

    display.set_style(
        relplot_kwargs={"color": ["red"]},
        perfect_model_kwargs={"color": "orange"},
    )
    display.plot()
    for line in display.lines_:
        assert line.get_color() == "orange"
    for ax in display.ax_:
        scatter_collection = ax.collections[0]
        np.testing.assert_array_equal(
            scatter_collection.get_facecolor()[0][:3], [1.0, 0.0, 0.0]
        )


def test_frame(linear_regression_with_train_test):
    """Test the frame method with comparison data."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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

    expected_index = ["estimator"]
    expected_columns = ["y_true", "y_pred", "residuals"]

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == 2


def test_legend(pyplot, linear_regression_with_train_test):
    """Check the rendering of the legend for prediction error with a
    `ComparisonReport`."""

    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts

    display.plot(kind="actual_vs_predicted")
    assert len(display.figure_.legends) == 1
    legend = display.figure_.legends[0]
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Perfect predictions" in legend_texts

    reports = {
        f"estimator {i}": EstimatorReport(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        for i in range(1, 10)
    }
    report = ComparisonReport(reports=reports)
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


def test_constructor(linear_regression_with_train_test):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.prediction_error()

    index_columns = ["estimator", "split"]
    df = display._prediction_error
    assert all(col in df.columns for col in index_columns)
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].isnull().all()


@pytest.mark.parametrize("subplot_by", [None, "estimator", "auto", "invalid"])
def test_subplot_by(pyplot, linear_regression_with_train_test, subplot_by):
    """Check that the subplot_by parameter works correctly for comparison reports."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    if subplot_by in ["invalid", None]:
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: auto, estimator. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    else:
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_[0], mpl.axes.Axes)
        assert len(display.ax_) == len(report.reports_)


def test_title(pyplot, linear_regression_with_train_test):
    """Check that the title contains expected elements."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    display.plot()
    title = display.figure_._suptitle.get_text()
    assert "Prediction Error" in title
    assert "estimator_1" not in title
    assert "Data source: Test set" in title


@pytest.fixture
def linear_regression_comparison_report(linear_regression_with_train_test):
    """Fixture providing a ComparisonReport with two linear regression estimators."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
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
    return report


def test_regression_data_source_both(pyplot, linear_regression_comparison_report):
    """Regression test: `data_source='both'` should plot without crashing."""

    report = linear_regression_comparison_report
    display = report.metrics.prediction_error(data_source="both")
    assert isinstance(display, PredictionErrorDisplay)

    display.plot()

    assert isinstance(display.ax_, (list, np.ndarray))
    assert len(display.ax_) == len(report.reports_)

    legend_texts = [text.get_text() for text in display.figure_.legends[0].get_texts()]
    assert len(legend_texts) == 2 + 1  # 2 datasource + 1 perfect predictions
    assert legend_texts[0] == "train"
    assert legend_texts[1] == "test"
