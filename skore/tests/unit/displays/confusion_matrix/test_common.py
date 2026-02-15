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
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
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
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_frame_structure(self, fixture_prefix, task, request):
        """Check that the frame method returns a properly structured dataframe."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.confusion_matrix()
        n_classes = len(display.display_labels)
        n_splits = 2 if "cross_validation" in fixture_prefix else 1
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
            2 if "cross_validation" in fixture_prefix else 0
        )

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_confusion_matrix_structure(self, fixture_prefix, task, request):
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
        expected_rows = 0
        for _, group in display.confusion_matrix.groupby(
            ["split", "estimator"], dropna=False
        ):
            n_t = max(1, group["threshold"].nunique())
            expected_rows += n_classes * n_classes * n_t
        assert display.confusion_matrix.shape[0] == expected_rows

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_facet_grid_kwargs(self, pyplot, fixture_prefix, task, request):
        """Check that we can override default facet grid kwargs."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        assert figure.get_figheight() == 6

        display.set_style(facet_grid_kwargs={"height": 8}).plot()
        assert display.figure_.get_figheight() == 8

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_heatmap_kwargs(self, pyplot, fixture_prefix, task, request):
        """Check that heatmap kwargs are applied correctly and can be changed."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        _, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        ax = ax[0] if isinstance(ax, np.ndarray) else ax

        assert ax.collections[0].get_cmap().name == "Blues"
        display.set_style(heatmap_kwargs={"cmap": "Reds"}).plot()
        ax = display.ax_[0] if isinstance(display.ax_, np.ndarray) else display.ax_
        assert ax.collections[0].get_cmap().name == "Reds"

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_plot_attributes(self, pyplot, fixture_prefix, task, request):
        """Check that the plot has correct attributes and labels."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        figure, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        assert "Confusion Matrix" in figure.get_suptitle()

        ax = ax[0] if isinstance(ax, np.ndarray) else ax
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

    def test_thresholds_available_for_binary_classification(
        self, fixture_prefix, request
    ):
        """Check that thresholds are available for binary classification."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert display.thresholds is not None
        assert len(display.thresholds) > 0
        assert "threshold" in display.confusion_matrix.columns

        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_binary_classification_figure_axes"
        )
        assert "threshold" in figure.get_suptitle().lower()

    def test_thresholds_in_multiclass(self, pyplot, fixture_prefix, request):
        """Check that the absence of thresholds in handled properly in multiclass."""
        report = request.getfixturevalue(f"{fixture_prefix}_multiclass_classification")
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

    def test_threshold_values_are_sorted(self, fixture_prefix, request):
        """Check that thresholds are sorted in ascending order."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert np.all(display.thresholds[:-1] <= display.thresholds[1:])

    def test_threshold_values_are_unique(self, fixture_prefix, request):
        """Check that thresholds contains unique values."""
        report = request.getfixturevalue(f"{fixture_prefix}_multiclass_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert len(display.thresholds) == len(np.unique(display.thresholds))

    def test_plot_multiclass_no_threshold_in_title(self, fixture_prefix, request):
        """Check that multiclass classification does not show threshold in title."""
        report = request.getfixturevalue(f"{fixture_prefix}_multiclass_classification")
        if isinstance(report, tuple):
            report = report[0]
        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_multiclass_classification_figure_axes"
        )

        expected_title = "Confusion Matrix" + "\nData source: Test set"
        assert figure.get_suptitle() == expected_title
        assert "threshold" not in figure.get_suptitle().lower()

    def test_threshold_greater_than_max(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        frame = display.frame(threshold_value=1.1)
        cm = display.confusion_matrix

        for (_est, _split), group in frame.groupby(["estimator", "split"]):
            assert group["threshold"].nunique() == 1
            mask_est = cm["estimator"] == _est
            mask_split = (
                cm["split"].isna() if pd.isna(_split) else cm["split"] == _split
            )
            full = cm[mask_est & mask_split]
            assert group["threshold"].iloc[0] == full["threshold"].max()

    def test_threshold_lower_than_min(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        frame = display.frame(threshold_value=-0.1)
        cm = display.confusion_matrix

        for (_est, _split), group in frame.groupby(["estimator", "split"]):
            assert group["threshold"].nunique() == 1
            mask_est = cm["estimator"] == _est
            mask_split = (
                cm["split"].isna() if pd.isna(_split) else cm["split"] == _split
            )
            full = cm[mask_est & mask_split]
            assert group["threshold"].iloc[0] == full["threshold"].min()

    def test_frame_none_returns_all_thresholds_binary(self, fixture_prefix, request):
        """Check that frame(threshold_value=None) returns all thresholds for binary."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        # Binary classification has real thresholds (no nan)
        n_thresholds = int(np.sum(~np.isnan(display.thresholds)))

        frame_all = display.frame(threshold_value=None)
        frame_default = display.frame(threshold_value="default")

        assert frame_all["threshold"].nunique() == n_thresholds
        assert frame_all.shape[0] == len(display.confusion_matrix)
        assert frame_all.shape[0] > frame_default.shape[0]

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_normalization(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        for normalize in ("true", "pred", "all"):
            frame = display.frame(normalize=normalize)
            for (_est, _split), group in frame.groupby(["estimator", "split"]):
                pivoted = group.pivot(
                    index="true_label",
                    columns="predicted_label",
                    values="value",
                )
                if normalize == "true":
                    row_sums = pivoted.sum(axis=1)
                    valid = (np.abs(row_sums - 1.0) < 1e-10) | (
                        np.abs(row_sums) < 1e-10
                    )
                    assert np.all(valid), f"row sums should be 0 or 1, got {row_sums}"
                elif normalize == "pred":
                    col_sums = pivoted.sum(axis=0)
                    valid = (np.abs(col_sums - 1.0) < 1e-10) | (
                        np.abs(col_sums) < 1e-10
                    )
                    assert np.all(valid), f"col sums should be 0 or 1, got {col_sums}"
                else:
                    np.testing.assert_allclose(pivoted.sum().sum(), 1.0)
