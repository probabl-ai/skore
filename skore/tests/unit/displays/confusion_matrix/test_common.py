import numpy as np
import pandas as pd
import pytest

from skore import EstimatorReport
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

        assert hasattr(display, "confusion_matrix_predict")
        assert hasattr(display, "confusion_matrix_thresholded")
        assert hasattr(display, "report_type")
        assert hasattr(display, "ml_task")
        assert hasattr(display, "data_source")
        assert hasattr(display, "report_pos_label")

        assert hasattr(display, "labels")

        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_frame_structure(self, fixture_prefix, task, request):
        """Check that the frame method returns a properly structured dataframe."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.confusion_matrix()
        n_classes = len(display.labels)
        n_splits = 2 if "cross_validation" in fixture_prefix else 1
        n_reports = 2 if "comparison" in fixture_prefix else 1

        frame = display.frame()
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (n_classes * n_classes * n_splits * n_reports, 6)

        expected_columns = [
            "true_label",
            "predicted_label",
            "value",
            "split",
            "estimator",
            "data_source",
        ]
        assert frame.columns.tolist() == expected_columns
        assert set(frame["true_label"]) == set(display.labels)
        assert set(frame["predicted_label"]) == set(display.labels)
        assert frame["split"].nunique() == (
            2 if "cross_validation" in fixture_prefix else 0
        )

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_confusion_matrix_predict_structure(self, fixture_prefix, task, request):
        """Check the structure of the confusion_matrix_predict attribute."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert isinstance(display.confusion_matrix_predict, pd.DataFrame)
        assert display.confusion_matrix_predict.columns.tolist() == [
            "true_label",
            "predicted_label",
            "count",
            "normalized_by_true",
            "normalized_by_pred",
            "normalized_by_all",
            "split",
            "estimator",
            "data_source",
        ]
        n_classes = len(display.labels)
        n_splits = 2 if "cross_validation" in fixture_prefix else 1
        n_reports = 2 if "comparison" in fixture_prefix else 1
        assert display.confusion_matrix_predict.shape[0] == (
            n_classes * n_classes * n_splits * n_reports
        )

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_confusion_matrix_thresholded_structure(
        self, fixture_prefix, task, request
    ):
        """Check the structure of the confusion_matrix_thresholded attribute."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert isinstance(display.confusion_matrix_thresholded, pd.DataFrame)
        assert display.confusion_matrix_thresholded.columns.tolist() == [
            "true_label",
            "predicted_label",
            "count",
            "normalized_by_true",
            "normalized_by_pred",
            "normalized_by_all",
            "threshold",
            "label",
            "split",
            "estimator",
            "data_source",
        ]
        expected_rows = 0
        for _, group in display.confusion_matrix_thresholded.groupby(
            ["split", "estimator"], dropna=False, observed=True
        ):
            n_t = max(1, group["threshold"].nunique())
            expected_rows += 2 * 2 * n_t
        assert display.confusion_matrix_thresholded.shape[0] == expected_rows

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

        display.set_style(facet_grid_kwargs={"height": 8})
        fig = display.plot()
        assert fig.get_figheight() == 8

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
        ax = ax[0]

        assert ax.collections[0].get_cmap().name == "Blues"
        display.set_style(heatmap_kwargs={"cmap": "Reds"})
        fig = display.plot()
        ax = fig.axes[0]
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

        ax = ax[0]
        assert ax.get_xlabel() == "Predicted label"
        assert ax.get_ylabel() == "True label"

        n_classes = len(display.labels)
        assert len(ax.get_xticks()) == n_classes
        assert len(ax.get_yticks()) == n_classes

        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]
        if task == "binary":
            assert xticklabels == ["0", "1"]
            assert yticklabels == ["0", "1"]
        else:
            assert xticklabels == ["0", "1", "2"]
            assert yticklabels == ["0", "1", "2"]

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_plot_no_threshold_in_title_by_default(self, fixture_prefix, task, request):
        """Check that default plot (predict-based) does not show threshold in title."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )

        expected_title = "Confusion Matrix" + "\nData source: Test set"
        assert figure.get_suptitle() == expected_title
        assert "threshold" not in figure.get_suptitle().lower()

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_frame_all_returns_all_thresholds(self, fixture_prefix, task, request):
        """Check that frame(threshold_value="all") returns all thresholds."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        frame_all = display.frame(threshold_value="all")
        assert frame_all.shape[0] == display.confusion_matrix_thresholded.shape[0]

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_normalization(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        for normalize in ("true", "pred", "all"):
            frame = display.frame(normalize=normalize)
            for (_est, _split), group in frame.groupby(
                ["estimator", "split"], observed=True
            ):
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

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_threshold_selection(self, fixture_prefix, task, request):
        """Check that frame snaps to the closest threshold and clips outside [0, 1]."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        label = display.labels[-1]
        frame_all = display.frame(threshold_value="all", label=label)

        groupby_cols = []
        if "cross-validation" in display.report_type:
            groupby_cols.append("split")
        if "comparison" in display.report_type:
            groupby_cols.append("estimator")

        def iter_groups(frame):
            if groupby_cols:
                return frame.groupby(groupby_cols, observed=True)
            return [("_ungrouped", frame)]

        available_thresholds_by_group = {
            key: group["threshold"].unique() for key, group in iter_groups(frame_all)
        }

        for test_value in [0.3, 0.7, -10.0, 10.0]:
            frame_selected = display.frame(threshold_value=test_value, label=label)
            for key, selected_group in iter_groups(frame_selected):
                selected_threshold = selected_group["threshold"].unique()
                assert len(selected_threshold) == 1
                available_thresholds = available_thresholds_by_group[key]
                expected_threshold = available_thresholds[
                    np.argmin(np.abs(available_thresholds - test_value))
                ]
                assert selected_threshold[0] == expected_threshold


def test_data_source_both_is_not_supported(forest_binary_classification_with_test):
    """Check that confusion_matrix rejects data_source='both' explicitly."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )

    with pytest.raises(
        ValueError,
        match="data_source='both' is not supported for confusion_matrix.",
    ):
        report.metrics.confusion_matrix(data_source="both")
