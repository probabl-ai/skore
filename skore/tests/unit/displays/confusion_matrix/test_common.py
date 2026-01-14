import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from skore._sklearn._plot import ConfusionMatrixDisplay


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
        "comparison_estimator_reports",
        "comparison_cross_validation_reports",
    ],
)
class TestConfusionMatrixDisplay:
    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_class_attributes(pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.confusion_matrix()
        assert isinstance(display, ConfusionMatrixDisplay)

        assert hasattr(display, "confusion_matrix")
        assert hasattr(display, "display_labels")
        assert hasattr(display, "report_type")
        assert hasattr(display, "ml_task")
        assert hasattr(display, "data_source")
        assert hasattr(display, "pos_label")
        assert hasattr(display, "response_method")

        assert hasattr(display, "thresholds")

        display.plot()
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_frame_structure(pyplot, fixture_prefix, task, request):
        """Check that the frame method returns a properly structured dataframe."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.confusion_matrix()
        n_classes = len(display.display_labels)
        n_splits = 5 if "cross_validation" in fixture_prefix else 1
        n_reports = 2 if "comparison" in fixture_prefix else 1

        frame = display.frame()
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (n_classes * n_classes * n_splits * n_reports, 7)

        expected_columns = [
            "true_label",
            "predicted_label",
            "value",
            "threshold",
            "split",
            "estimator",
            "data_source",
        ]
        assert frame.columns.tolist() == expected_columns
        assert set(frame["true_label"]) == set(display.display_labels)
        assert set(frame["predicted_label"]) == set(display.display_labels)
        assert frame["split"].nunique() == (
            5 if "cross_validation" in fixture_prefix else 0
        )

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_confusion_matrix_structure(pyplot, fixture_prefix, task, request):
        """Check the structure of the confusion_matrix attribute."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert isinstance(display.confusion_matrix, pd.DataFrame)
        assert display.confusion_matrix.columns.tolist() == [
            "true_label",
            "predicted_label",
            "count",
            "normalized_by_true",
            "normalized_by_pred",
            "normalized_by_all",
            "threshold",
            "split",
            "estimator",
            "data_source",
        ]
        n_classes = len(display.display_labels)
        n_thresholds = len(display.thresholds)
        n_splits = 5 if "cross_validation" in fixture_prefix else 1
        n_reports = 2 if "comparison" in fixture_prefix else 1
        expected_rows = n_thresholds * n_classes * n_classes * n_splits * n_reports
        assert display.confusion_matrix.shape[0] == expected_rows

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_facet_grid_kwargs(pyplot, fixture_prefix, task, request):
        """Check that we can override default facet grid kwargs."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        display.plot()
        assert display.figure_.get_figheight() == 6

        display.plot(facet_grid_kwargs={"height": 8})
        assert display.figure_.get_figheight() == 8

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_heatmap_kwargs(pyplot, fixture_prefix, task, request):
        """Check that heatmap kwargs are applied correctly and can be changed."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        def get_ax(display):
            return (
                display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_
            )

        n_base_elements = 1 if task == "binary" else 0
        n_plots = 2 if "comparison" in fixture_prefix else 1

        display = report.metrics.confusion_matrix()
        display.plot()
        assert get_ax(display).collections[0].get_cmap().name == "Blues"
        display.plot(heatmap_kwargs={"cmap": "Reds"})
        assert get_ax(display).collections[0].get_cmap().name == "Reds"
        display.set_style(heatmap_kwargs={"cmap": "Greens"}, policy="update").plot()
        assert get_ax(display).collections[0].get_cmap().name == "Greens"

        display = report.metrics.confusion_matrix()
        display.plot()
        assert len(get_ax(display).texts) > 1
        display.plot(heatmap_kwargs={"annot": False})
        # There is still the pos_label annotation
        assert len(get_ax(display).texts) == n_base_elements
        plt.close("all")

        display = report.metrics.confusion_matrix()
        display.plot(normalize="all")
        for text in get_ax(display).texts:
            text_content = text.get_text()
            assert "." in text_content or "*" in text_content
        display.plot(normalize="all", heatmap_kwargs={"fmt": ".2e"})
        for text in get_ax(display).texts:
            text_content = text.get_text()
            assert "e" in text_content
        display.set_style(heatmap_kwargs={"fmt": ".2E"}, policy="update").plot(
            normalize="all"
        )
        for text in get_ax(display).texts:
            text_content = text.get_text()
            assert "E" in text_content or "*" in text_content
        plt.close("all")

        display = report.metrics.confusion_matrix()
        display.plot()
        assert len(display.figure_.axes) == n_plots
        display.plot(heatmap_kwargs={"cbar": True})
        assert len(display.figure_.axes) == 2 * n_plots
        display.set_style(heatmap_kwargs={"cbar": False}, policy="update").plot()
        assert len(display.figure_.axes) == n_plots
        plt.close("all")

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_plot_attributes(pyplot, fixture_prefix, task, request):
        """Check that the plot has correct attributes and labels."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        display.plot()
        assert "Confusion Matrix" in display.figure_.get_suptitle()

        ax = display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_
        assert ax.get_xlabel() == "Predicted label"
        assert ax.get_ylabel() == "True label"

        n_classes = len(display.display_labels)
        assert len(ax.get_xticks()) == n_classes
        assert len(ax.get_yticks()) == n_classes

        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        if task == "binary":
            assert xticklabels == ["0", "1*"]
            assert yticklabels == ["0", "1*"]
        else:
            assert xticklabels == ["0", "1", "2"]
            assert yticklabels == ["0", "1", "2"]

    @pytest.mark.parametrize("task", ["binary"])
    def test_thresholds_available_for_binary_classification(
        pyplot, fixture_prefix, task, request
    ):
        """Check that thresholds are available for binary classification."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert display.thresholds is not None
        assert len(display.thresholds) > 0
        assert "threshold" in display.confusion_matrix.columns

        display.plot()
        assert "threshold" in display.figure_.get_suptitle().lower()

    @pytest.mark.parametrize("task", ["multiclass"])
    def test_thresholds_in_multiclass(pyplot, fixture_prefix, task, request):
        """Check that the absence of thresholds in handled properly in multiclass."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert len(display.thresholds) == 1
        assert np.isnan(display.thresholds[0])

        err_msg = "Threshold support is only available for binary classification."
        with pytest.raises(ValueError, match=err_msg):
            display.frame(threshold_value=0.5)
        with pytest.raises(ValueError, match=err_msg):
            display.plot(threshold_value=0.5)

    @pytest.mark.parametrize("task", ["binary"])
    def test_threshold_values_are_sorted(pyplot, fixture_prefix, task, request):
        """Check that thresholds are sorted in ascending order."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert np.all(display.thresholds[:-1] <= display.thresholds[1:])

    @pytest.mark.parametrize("task", ["binary"])
    def test_threshold_values_are_unique(pyplot, fixture_prefix, task, request):
        """Check that thresholds contains unique values."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert len(display.thresholds) == len(np.unique(display.thresholds))

    @pytest.mark.parametrize("task", ["multiclass"])
    def test_plot_multiclass_no_threshold_in_title(
        pyplot, fixture_prefix, task, request
    ):
        """Check that multiclass classification does not show threshold in title."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        display.plot()

        expected_title = "Confusion Matrix" + "\nData source: Test set"
        assert display.figure_.get_suptitle() == expected_title
        assert "threshold" not in display.figure_.get_suptitle().lower()
