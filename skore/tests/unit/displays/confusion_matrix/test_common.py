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

        assert hasattr(display, "thresholds")
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

    def test_thresholds_available_for_binary_classification(
        self, fixture_prefix, request
    ):
        """Check that thresholds are available for binary classification."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert display.confusion_matrix_thresholded is not None
        assert display.thresholds is not None
        assert len(display.thresholds) > 0

    def test_thresholds_in_multiclass(self, pyplot, fixture_prefix, request):
        """Check that the absence of thresholds is handled properly in multiclass."""
        report = request.getfixturevalue(f"{fixture_prefix}_multiclass_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert display.confusion_matrix_thresholded is not None
        assert len(display.thresholds) > 0

    def test_threshold_values_are_sorted(self, fixture_prefix, request):
        """Check that thresholds are sorted in ascending order."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert np.all(display.thresholds[:-1] <= display.thresholds[1:])

    def test_threshold_values_are_unique(self, fixture_prefix, request):
        """Check that thresholds contains unique values."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        assert len(display.thresholds) == len(np.unique(display.thresholds))

    def test_plot_no_threshold_in_title_by_default(self, fixture_prefix, request):
        """Check that default plot (predict-based) does not show threshold in title."""
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
        label = display.labels[-1]
        frame = display.frame(threshold_value=1.1, label=label)
        frame_all = display.frame(threshold_value="all", label=label)

        for (_est, _split), group in frame.groupby(
            ["estimator", "split"], observed=True
        ):
            assert group["threshold"].nunique() == 1
            group_all = frame_all
            if _est is not None and not (isinstance(_est, float) and np.isnan(_est)):
                group_all = group_all.query(f"estimator == '{_est}'")
            if _split is not None and not (
                isinstance(_split, float) and np.isnan(_split)
            ):
                group_all = group_all.query(f"split == {_split}")
            assert group["threshold"].iloc[0] == group_all["threshold"].max()

    def test_threshold_lower_than_min(self, fixture_prefix, request):
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()
        label = display.labels[-1]
        frame = display.frame(threshold_value=-0.1, label=label)
        frame_all = display.frame(threshold_value="all", label=label)

        for (_est, _split), group in frame.groupby(
            ["estimator", "split"], observed=True
        ):
            assert group["threshold"].nunique() == 1
            group_all = frame_all
            if _est is not None and not (isinstance(_est, float) and np.isnan(_est)):
                group_all = group_all.query(f"estimator == '{_est}'")
            if _split is not None and not (
                isinstance(_split, float) and np.isnan(_split)
            ):
                group_all = group_all.query(f"split == {_split}")
            assert group["threshold"].iloc[0] == group_all["threshold"].min()

    def test_frame_all_returns_all_thresholds_binary(self, fixture_prefix, request):
        """Check that frame(threshold_value="all") returns all thresholds for binary."""
        report = request.getfixturevalue(f"{fixture_prefix}_binary_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.confusion_matrix()

        frame_all = display.frame(threshold_value="all")
        frame_at_threshold = display.frame(threshold_value=0.5)

        assert frame_all["threshold"].nunique() > 1
        assert frame_all.shape[0] > frame_at_threshold.shape[0]

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
