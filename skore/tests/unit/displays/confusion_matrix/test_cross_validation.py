import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import CrossValidationReport
from skore._sklearn._plot import ConfusionMatrixDisplay


def test_binary_classification(pyplot, cross_validation_report_binary_classification):
    """Check that the confusion matrix display is correct for binary classification."""
    report = cross_validation_report_binary_classification

    display = report.metrics.confusion_matrix()
    display.plot()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert hasattr(display, "confusion_matrix")
    assert hasattr(display, "display_labels")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert len(display.display_labels) == 2
    assert "Confusion Matrix" in display.ax_.get_title()
    assert "Decision threshold" in display.ax_.get_title()
    assert display.ax_.get_xlabel() == "Predicted label"
    assert display.ax_.get_ylabel() == "True label"

    n_classes = len(display.display_labels)
    # There can be thresholds that are equal in different splits so
    # sum(n_thresholds_per_split) != len(display.thresholds)
    n_thresholds_per_split = display.confusion_matrix.groupby("split")[
        "threshold"
    ].nunique()
    expected_rows = (n_thresholds_per_split * n_classes * n_classes).sum()
    assert display.confusion_matrix.shape == (expected_rows, 8)


def test_multiclass_classification(
    pyplot, cross_validation_report_multiclass_classification
):
    """Check that the confusion matrix display is correct for multiclass
    classification.
    """
    report = cross_validation_report_multiclass_classification

    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    n_classes = len(display.display_labels)
    assert len(display.display_labels) == n_classes
    assert display.confusion_matrix.shape == (n_classes * n_classes * 3, 8)


def test_confusion_matrix(pyplot, cross_validation_report_binary_classification):
    """Check the structure of the confusion_matrix attribute."""
    report = cross_validation_report_binary_classification
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
    ]
    n_classes = len(display.display_labels)
    n_thresholds_per_split = display.confusion_matrix.groupby("split")[
        "threshold"
    ].nunique()
    expected_rows = (n_thresholds_per_split * n_classes * n_classes).sum()
    assert display.confusion_matrix.shape[0] == expected_rows
    assert display.confusion_matrix["split"].nunique() == 3


def test_normalization(pyplot, cross_validation_report_binary_classification):
    """Check that normalized values are correctly computed."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    threshold_val = display.thresholds[len(display.thresholds) // 2]

    for split_idx in range(3):
        cm_true = display.frame(threshold_value=threshold_val, normalize="true").query(
            f"split == {split_idx}"
        )
        pivoted_true = cm_true.pivot(
            index="true_label", columns="predicted_label", values="value"
        )
        np.testing.assert_allclose(pivoted_true.sum(axis=1), 1.0)

        cm_pred = display.frame(threshold_value=threshold_val, normalize="pred").query(
            f"split == {split_idx}"
        )
        pivoted_pred = cm_pred.pivot(
            index="true_label", columns="predicted_label", values="value"
        )
        np.testing.assert_allclose(pivoted_pred.sum(axis=0), 1.0)

        cm_all = display.frame(threshold_value=threshold_val, normalize="all").query(
            f"split == {split_idx}"
        )
        pivoted_all = cm_all.pivot(
            index="true_label", columns="predicted_label", values="value"
        )
        np.testing.assert_allclose(pivoted_all.sum().sum(), 1.0)


def test_data_source(pyplot, forest_binary_classification_data):
    """Check that we can request the confusion matrix display with different data
    sources.
    """
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)

    display_test = report.metrics.confusion_matrix()
    display_train = report.metrics.confusion_matrix(data_source="train")
    display_custom = report.metrics.confusion_matrix(
        data_source="X_y",
        X=X,
        y=y,
    )

    assert display_test.confusion_matrix is not None
    assert display_train.confusion_matrix is not None
    assert display_custom.confusion_matrix is not None

    assert not np.array_equal(
        display_test.confusion_matrix["count"].values,
        display_train.confusion_matrix["count"].values,
    )

    assert not np.array_equal(
        display_test.confusion_matrix["count"].values,
        display_custom.confusion_matrix["count"].values,
    )

    assert display_test.confusion_matrix.shape[0] > 0
    assert display_train.confusion_matrix.shape[0] > 0
    assert display_custom.confusion_matrix.shape[0] > 0


def test_default_heatmap_kwargs(pyplot, cross_validation_report_binary_classification):
    """Check that default heatmap kwargs are applied correctly."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    display.plot()

    assert display.ax_.collections[0].get_cmap().name == "Blues"
    assert len(display.ax_.texts) > 0
    assert all(isinstance(text, mpl.text.Text) for text in display.ax_.texts)
    for text in display.ax_.texts:
        text_content = text.get_text()
        assert "±" in text_content or "*" in text_content
    assert len(display.figure_.axes) == 1


def test_heatmap_kwargs_override(pyplot, cross_validation_report_binary_classification):
    """Check that we can override default heatmap kwargs."""
    report = cross_validation_report_binary_classification

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"cmap": "Reds"})
    assert display.ax_.collections[0].get_cmap().name == "Reds"

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"annot": False})
    # There is still the pos_label annotation
    assert len(display.ax_.texts) == 1

    display = report.metrics.confusion_matrix()
    display.plot(normalize="all", heatmap_kwargs={"fmt": ".2e"})
    for text in display.ax_.texts:
        text_content = text.get_text()
        assert "e-" in text_content or "*" in text_content

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"cbar": True})
    assert len(display.figure_.axes) == 2


def test_set_style(pyplot, cross_validation_report_binary_classification):
    """Check that the set_style method works with heatmap_kwargs."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    display.set_style(heatmap_kwargs={"alpha": 0.5})
    display.plot()

    assert display.ax_.collections[0].get_alpha() == 0.5

    display.plot(heatmap_kwargs={"alpha": 0.8})
    assert display.ax_.collections[0].get_alpha() == 0.8


def test_plot_attributes(pyplot, cross_validation_report_binary_classification):
    """Check that the plot has correct attributes and labels."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    display.plot()

    assert display.ax_.get_xlabel() == "Predicted label"
    assert display.ax_.get_ylabel() == "True label"
    assert "Confusion Matrix" in display.ax_.get_title()

    n_classes = len(display.display_labels)
    assert len(display.ax_.get_xticks()) == n_classes
    assert len(display.ax_.get_yticks()) == n_classes

    xticklabels = [label.get_text() for label in display.ax_.get_xticklabels()]
    yticklabels = [label.get_text() for label in display.ax_.get_yticklabels()]
    assert xticklabels == ["0", "1*"]
    assert yticklabels == ["0", "1*"]


def test_frame_structure(cross_validation_report_binary_classification):
    """Check that the frame method returns a properly structured dataframe."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    n_classes = len(display.display_labels)

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes * n_classes * 3, 5)

    expected_columns = [
        "true_label",
        "predicted_label",
        "value",
        "threshold",
        "split",
    ]
    assert frame.columns.tolist() == expected_columns
    assert set(frame["true_label"].unique()) == set(display.display_labels)
    assert set(frame["predicted_label"].unique()) == set(display.display_labels)
    assert frame["split"].nunique() == 3


def test_thresholds_available_for_binary_classification(
    pyplot, cross_validation_report_binary_classification
):
    """Check that thresholds are available for binary classification."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert display.thresholds is not None
    assert len(display.thresholds) > 0
    assert "threshold" in display.confusion_matrix.columns


def test_thresholds_not_available_for_multiclass(
    pyplot, cross_validation_report_multiclass_classification
):
    """Check that thresholds are NaN for multiclass classification."""
    report = cross_validation_report_multiclass_classification
    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert len(display.thresholds) == 1
    assert np.isnan(display.thresholds[0])


def test_threshold_value_error_for_multiclass(
    cross_validation_report_multiclass_classification,
):
    """Check that specifying threshold_value for multiclass raises an error."""
    report = cross_validation_report_multiclass_classification
    display = report.metrics.confusion_matrix()

    err_msg = "Threshold support is only available for binary classification."
    with pytest.raises(ValueError, match=err_msg):
        display.frame(threshold_value=0.5)


def test_plot_with_threshold(pyplot, cross_validation_report_binary_classification):
    """Check that we can plot with a specific threshold."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    display.plot(threshold_value=0.3)
    assert "threshold" in display.ax_.get_title().lower()


def test_frame_with_threshold(cross_validation_report_binary_classification):
    """Check that we can get a frame at a specific threshold."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    frame = display.frame(threshold_value=0.5)

    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes * 3, 5)


@pytest.mark.parametrize(
    "estimator, expected_default_threshold",
    [(SVC(probability=False), 0), (LogisticRegression(), 0.5)],
)
def test_frame_default_threshold(
    binary_classification_data, estimator, expected_default_threshold
):
    """Check that frame() uses default threshold."""
    X, y = binary_classification_data
    cv = 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

    frame = display.frame(threshold_value=None)
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes * cv, 5)

    for split_idx in range(cv):
        split_frame = frame.query(f"split == {split_idx}")
        assert split_frame["threshold"].nunique() == 1

        split_thresholds = display.confusion_matrix.query(f"split == {split_idx}")[
            "threshold"
        ].unique()
        closest_threshold = split_thresholds[
            np.argmin(abs(split_thresholds - expected_default_threshold))
        ]
        assert split_frame["threshold"].unique()[0] == closest_threshold


def test_threshold_closest_match(pyplot, cross_validation_report_binary_classification):
    """Check that the closest threshold is selected for data."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    middle_index = len(display.thresholds) // 2
    threshold = (
        display.thresholds[middle_index] + display.thresholds[middle_index + 1]
    ) / 2 - 1e-6
    assert threshold not in display.thresholds

    display.plot(threshold_value=threshold)
    expected_title = f"Confusion Matrix\nDecision threshold: {threshold:.2f}"
    assert display.ax_.get_title() == expected_title

    frame = display.frame(normalize=None, threshold_value=threshold)
    aggregated = (
        frame.groupby(["true_label", "predicted_label"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    expected_values = aggregated.pivot(
        index="true_label", columns="predicted_label", values="mean"
    ).reindex(index=display.display_labels, columns=display.display_labels)

    np.testing.assert_allclose(
        display.ax_.collections[0].get_array(),
        expected_values.values,
    )


def test_pos_label(pyplot, forest_binary_classification_data):
    """Check that the pos_label parameter works correctly."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    estimator = RandomForestClassifier()
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)

    display = report.metrics.confusion_matrix(pos_label="A")
    display.plot()
    assert display.ax_.get_xticklabels()[1].get_text() == "A*"
    assert display.ax_.get_yticklabels()[1].get_text() == "A*"

    display = report.metrics.confusion_matrix(pos_label="B")
    display.plot()
    assert display.ax_.get_xticklabels()[1].get_text() == "B*"
    assert display.ax_.get_yticklabels()[1].get_text() == "B*"


def test_plot_multiclass_no_threshold_in_title(
    pyplot, cross_validation_report_multiclass_classification
):
    """Check that multiclass classification does not show threshold in title."""
    report = cross_validation_report_multiclass_classification
    display = report.metrics.confusion_matrix()
    display.plot()

    assert display.ax_.get_title() == "Confusion Matrix"
    assert "threshold" not in display.ax_.get_title().lower()


def test_multiple_thresholds_different_confusion_matrices(
    cross_validation_report_binary_classification,
):
    """Check that different thresholds produce different confusion matrices."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    low_threshold = display.thresholds[len(display.thresholds) // 4]
    high_threshold = display.thresholds[3 * len(display.thresholds) // 4]

    frame_low = display.frame(threshold_value=low_threshold)
    frame_high = display.frame(threshold_value=high_threshold)

    assert frame_low.shape == frame_high.shape
    assert frame_low["threshold"].iloc[0] != frame_high["threshold"].iloc[0]
    assert not np.array_equal(frame_low["value"].values, frame_high["value"].values)


def test_threshold_values_are_sorted(cross_validation_report_binary_classification):
    """Check that thresholds are sorted in ascending order."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    assert np.all(display.thresholds[:-1] <= display.thresholds[1:])


def test_threshold_values_are_unique(cross_validation_report_binary_classification):
    """Check that thresholds contains unique values."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    assert len(display.thresholds) == len(np.unique(display.thresholds))


def test_threshold_greater_than_max(cross_validation_report_binary_classification):
    """Check that a threshold greater than the maximum threshold is set to the maximum
    threshold."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    frame = display.frame(threshold_value=1.1)

    for split_idx in range(3):
        split_frame = frame.query(f"split == {split_idx}")
        assert split_frame["threshold"].nunique() == 1

        split_thresholds = display.confusion_matrix.query(f"split == {split_idx}")[
            "threshold"
        ].unique()
        split_max_threshold = np.max(split_thresholds)
        assert split_frame["threshold"].unique()[0] == split_max_threshold


def test_split_aggregation(pyplot, cross_validation_report_binary_classification):
    """Check that confusion matrix values are aggregated across splits."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()
    display.plot()

    annotation_texts = [
        text.get_text()
        for text in display.ax_.texts
        if text.get_text() and "±" in text.get_text()
    ]
    assert len(annotation_texts) == len(display.display_labels) ** 2

    for text_content in annotation_texts:
        assert "\n" in text_content
        assert "±" in text_content
        parts = text_content.split("\n")
        assert len(parts) == 2
        assert parts[1].startswith("(±")
        assert parts[1].endswith(")")


def test_split_column_present(cross_validation_report_binary_classification):
    """Check that the split column is present in confusion matrix dataframe."""
    report = cross_validation_report_binary_classification
    display = report.metrics.confusion_matrix()

    assert "split" in display.confusion_matrix.columns
    assert display.confusion_matrix["split"].nunique() == 3
    assert set(display.confusion_matrix["split"].unique()) == set(range(3))
