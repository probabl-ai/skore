"""Common tests for RocCurveDisplay."""

import numpy as np
import pytest
import seaborn as sns

from skore._sklearn._plot import RocCurveDisplay
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
class TestRocCurveDisplay:
    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.roc()
        assert isinstance(display, RocCurveDisplay)

        assert hasattr(display, "roc_curve")
        assert hasattr(display, "roc_auc")
        assert hasattr(display, "report_type")
        assert hasattr(display, "ml_task")
        assert hasattr(display, "data_source")
        assert hasattr(display, "pos_label")

        display.plot()
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    @pytest.mark.parametrize("with_roc_auc", [False, True])
    def test_frame_structure(self, fixture_prefix, task, with_roc_auc, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.roc()
        frame = display.frame(with_roc_auc=with_roc_auc)

        expected_columns = ["threshold", "fpr", "tpr"]
        expected_index = []
        if with_roc_auc:
            expected_columns.append("roc_auc")
        if "cross_validation" in fixture_prefix:
            expected_index.append("split")
        if "comparison" in fixture_prefix:
            expected_index.append("estimator")
        if task == "multiclass":
            expected_index.append("label")

        check_frame_structure(frame, expected_index, expected_columns)

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_internal_data_structure(self, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.roc()

        assert list(display.roc_curve.columns) == [
            "estimator",
            "data_source",
            "split",
            "label",
            "threshold",
            "fpr",
            "tpr",
        ]
        assert list(display.roc_auc.columns) == [
            "estimator",
            "data_source",
            "split",
            "label",
            "roc_auc",
        ]

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_relplot_kwargs(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.roc()

        _, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        ax = ax[0] if isinstance(ax, np.ndarray) else ax
        assert ax.get_lines()[0].get_color() == sns.color_palette()[0]
        relplot_kwargs = (
            {"palette": ["red", "green", "blue"]}
            if task == "multiclass"
            else {"color": "red"}
        )

        display.set_style(relplot_kwargs=relplot_kwargs).plot()
        ax = display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_
        assert ax.get_lines()[0].get_color() == "red"

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        _, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        ax = ax[0] if isinstance(ax, np.ndarray) else ax

        n_splits = 2 if "cross_validation" in fixture_prefix else 1
        n_labels = 3 if task == "multiclass" else 1
        n_lines = n_splits * n_labels + 1
        assert len(ax.get_lines()) == n_lines

        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() in ("True Positive Rate", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.roc()
        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        title = figure.get_suptitle()

        assert "ROC Curve" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display.roc_curve["estimator"].cat.categories[0]
            assert estimator_name in title
        else:
            assert "for" not in title
        if display.data_source in ("train", "test", "X_y"):
            assert "Data source" in title
        else:
            assert "Data source" not in title
        if task == "binary" and display.pos_label is not None:
            assert "Positive label" in title
        else:
            assert "Positive label" not in title
