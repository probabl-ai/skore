"""Common tests for PredictionErrorDisplay."""

import numpy as np
import pytest
import seaborn as sns

from skore._sklearn._plot import PredictionErrorDisplay
from skore._utils._testing import check_frame_structure


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
        "comparison_estimator_reports",
        "comparison_cross_validation_reports",
    ],
)
class TestPredictionErrorDisplay:
    def test_class_attributes(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.prediction_error()
        assert isinstance(display, PredictionErrorDisplay)

        assert hasattr(display, "_prediction_error")
        assert hasattr(display, "range_y_true")
        assert hasattr(display, "range_y_pred")
        assert hasattr(display, "range_residuals")
        assert hasattr(display, "report_type")
        assert hasattr(display, "ml_task")
        assert hasattr(display, "data_source")

        display.plot()
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    def test_frame_structure(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.prediction_error()
        frame = display.frame()

        expected_columns = ["y_true", "y_pred", "residuals"]
        expected_index = []
        if "cross_validation" in fixture_prefix:
            expected_index.append("split")
        if "comparison" in fixture_prefix:
            expected_index.append("estimator")

        check_frame_structure(frame, expected_index, expected_columns)

    def test_internal_data_structure(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.prediction_error()

        assert list(display._prediction_error.columns) == [
            "estimator",
            "data_source",
            "split",
            "y_true",
            "y_pred",
            "residuals",
        ]

    def test_relplot_kwargs(self, pyplot, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.prediction_error()
        _, ax = request.getfixturevalue(f"{fixture_prefix}_regression_figure_axes")
        ax = ax[0] if isinstance(ax, np.ndarray) else ax
        np.testing.assert_array_equal(
            ax.collections[0].get_facecolor()[0][:3], sns.color_palette()[0]
        )

        relplot_kwargs = (
            {"color": "red"}
            if "estimator_reports" in fixture_prefix
            else {"palette": ["red", "blue"]}
        )
        display.set_style(
            relplot_kwargs=relplot_kwargs,
            perfect_model_kwargs={"color": "blue"},
        ).plot()
        ax = display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_
        assert display.lines_[0].get_color() == "blue"
        np.testing.assert_array_equal(
            ax.collections[0].get_facecolor()[0][:3], [1.0, 0.0, 0.0]
        )

    def test_plot_structure(self, pyplot, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]
        _, ax = request.getfixturevalue(f"{fixture_prefix}_regression_figure_axes")
        ax = ax[0] if isinstance(ax, np.ndarray) else ax

        assert len(ax.get_lines()) >= 1
        assert ax.get_xlabel() == "Predicted values"
        assert ax.get_ylabel() == "Residuals (actual - predicted)"

    def test_title(self, pyplot, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_regression")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.prediction_error()
        figure, _ = request.getfixturevalue(f"{fixture_prefix}_regression_figure_axes")
        title = figure.get_suptitle()

        assert "Prediction Error" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display._prediction_error["estimator"].cat.categories[0]
            assert estimator_name in title
        else:
            assert "for" not in title
        if display.data_source in ("train", "test", "X_y"):
            assert "Data source" in title
        else:
            assert "Data source" not in title
