import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import EstimatorReport
from skore._sklearn._plot import ConfusionMatrixDisplay


def test_binary_classification(pyplot, forest_binary_classification_with_train_test):
    """Check that the confusion matrix display is correct for binary classification."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()
    display.plot()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert hasattr(display, "confusion_matrix")
    assert hasattr(display, "display_labels")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert len(display.display_labels) == 2
    assert "Confusion Matrix" in display.figure_.get_suptitle()
    assert "Decision threshold" in display.figure_.get_suptitle()
    assert display.ax_.get_xlabel() == "Predicted label"
    assert display.ax_.get_ylabel() == "True label"

    n_classes = len(display.display_labels)
    n_thresholds = len(display.thresholds)
    assert display.confusion_matrix.shape == (n_thresholds * n_classes * n_classes, 10)


def test_multiclass_classification(
    pyplot, forest_multiclass_classification_with_train_test
):
    """Check that the confusion matrix display is correct for multiclass
    classification.
    """
    estimator, X_train, X_test, y_train, y_test = (
        forest_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    n_classes = len(np.unique(y_test))
    assert len(display.display_labels) == n_classes
    assert display.confusion_matrix.shape == (n_classes * n_classes, 10)


def test_confusion_matrix(pyplot, forest_binary_classification_with_train_test):
    """Check the structure of the confusion_matrix attribute."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
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
    assert display.confusion_matrix.shape[0] == (n_thresholds * n_classes * n_classes)


def test_normalization(pyplot, forest_binary_classification_with_train_test):
    """Check that normalized values are correctly computed."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()
    cm = display.confusion_matrix
    threshold_val = display.thresholds[len(display.thresholds) // 2]
    cm_at_threshold = cm[cm["threshold"] == threshold_val]

    pivoted_true = cm_at_threshold.pivot(
        index="true_label", columns="predicted_label", values="normalized_by_true"
    )
    np.testing.assert_allclose(pivoted_true.sum(axis=1), 1.0)

    pivoted_pred = cm_at_threshold.pivot(
        index="true_label", columns="predicted_label", values="normalized_by_pred"
    )
    np.testing.assert_allclose(pivoted_pred.sum(axis=0), 1.0)

    pivoted_all = cm_at_threshold.pivot(
        index="true_label", columns="predicted_label", values="normalized_by_all"
    )
    np.testing.assert_allclose(pivoted_all.sum().sum(), 1.0)


def test_data_source(pyplot, forest_binary_classification_with_train_test):
    """Check that we can request the confusion matrix display with different data
    sources.
    """
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display_test = report.metrics.confusion_matrix()
    display_train = report.metrics.confusion_matrix(data_source="train")
    display_custom = report.metrics.confusion_matrix(
        data_source="X_y",
        X=X_train,
        y=y_train,
    )

    assert display_test.confusion_matrix is not None
    assert not np.array_equal(
        display_test.confusion_matrix["count"].values,
        display_train.confusion_matrix["count"].values,
    )

    assert np.array_equal(
        display_train.confusion_matrix["count"].values,
        display_custom.confusion_matrix["count"].values,
    )


def test_default_heatmap_kwargs(pyplot, forest_binary_classification_with_train_test):
    """Check that default heatmap kwargs are applied correctly."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()
    display.plot()

    assert display.ax_.collections[0].get_cmap().name == "Blues"
    assert len(display.ax_.texts) > 0
    assert all(isinstance(text, mpl.text.Text) for text in display.ax_.texts)
    for text in display.ax_.texts:
        assert "." not in text.get_text()
    assert len(display.figure_.axes) == 1


def test_heatmap_kwargs_override(pyplot, forest_binary_classification_with_train_test):
    """Check that we can override default heatmap kwargs."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"cmap": "Reds"})
    assert display.ax_.collections[0].get_cmap().name == "Reds"

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"annot": False})
    # There is still the pos_label annotation
    assert len(display.ax_.texts) == 1

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"fmt": "d"})
    for text in display.ax_.texts:
        assert "." not in text.get_text() or text.get_text().endswith(".0")

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"cbar": True})
    assert len(display.figure_.axes) == 2
    display.plot(heatmap_kwargs={"cbar": True})
    assert len(display.figure_.axes) == 2


def test_set_style(pyplot, forest_binary_classification_with_train_test):
    """Check that the set_style method works with heatmap_kwargs."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()

    display.set_style(heatmap_kwargs={"alpha": 0.5})
    display.plot()

    assert display.ax_.collections[0].get_alpha() == 0.5

    display.plot(heatmap_kwargs={"alpha": 0.8})
    assert display.ax_.collections[0].get_alpha() == 0.8


def test_plot_attributes(pyplot, forest_binary_classification_with_train_test):
    """Check that the plot has correct attributes and labels."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()
    display.plot()

    assert display.ax_.get_xlabel() == "Predicted label"
    assert display.ax_.get_ylabel() == "True label"
    assert "Confusion Matrix" in display.figure_.get_suptitle()

    n_classes = len(display.display_labels)
    assert len(display.ax_.get_xticks()) == n_classes
    assert len(display.ax_.get_yticks()) == n_classes

    xticklabels = [label.get_text() for label in display.ax_.get_xticklabels()]
    yticklabels = [label.get_text() for label in display.ax_.get_yticklabels()]
    assert xticklabels == ["0", "1*"]
    assert yticklabels == ["0", "1*"]
    assert xticklabels == ["0", "1*"]
    assert yticklabels == ["0", "1*"]


def test_frame_structure(forest_binary_classification_with_train_test):
    """Check that the frame method returns a properly structured dataframe."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()
    n_classes = len(display.display_labels)

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes * n_classes, 7)

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
    assert set(frame["true_label"].unique()) == set(display.display_labels)
    assert set(frame["predicted_label"].unique()) == set(display.display_labels)


def test_thresholdsavailable_for_binary_classification(
    pyplot, forest_binary_classification_with_train_test
):
    """Check that thresholds are available for binary classification."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert display.thresholds is not None
    assert len(display.thresholds) > 0
    assert "threshold" in display.confusion_matrix.columns


def test_thresholdsnot_available_for_multiclass(
    pyplot, forest_multiclass_classification_with_train_test
):
    """Check that thresholds are NaN for multiclass classification."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    assert isinstance(display, ConfusionMatrixDisplay)
    assert len(display.thresholds) == 1
    assert np.isnan(display.thresholds[0])


def test_threshold_value_error_for_multiclass(
    forest_multiclass_classification_with_train_test,
):
    """Check that specifying threshold_value for multiclass raises an error."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    err_msg = "Threshold support is only available for binary classification."
    with pytest.raises(ValueError, match=err_msg):
        display.frame(threshold_value=0.5)


def test_plot_with_threshold(pyplot, forest_binary_classification_with_train_test):
    """Check that we can plot with a specific threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    display.plot(threshold_value=0.3)
    assert "threshold" in display.figure_.get_suptitle().lower()


def test_frame_with_threshold(forest_binary_classification_with_train_test):
    """Check that we can get a frame at a specific threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()
    frame = display.frame(threshold_value=0.5)

    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes, 7)


@pytest.mark.parametrize(
    "estimator, expected_default_threshold",
    [(SVC(probability=False), 0), (LogisticRegression(), 0.5)],
)
def test_frame_default_threshold(
    binary_classification_train_test_split, estimator, expected_default_threshold
):
    """Check that frame() uses default threshold."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    frame = display.frame(threshold_value=None)
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes, 7)
    assert frame["threshold"].nunique() == 1
    closest_threshold = display.thresholds[
        np.argmin(abs(display.thresholds - expected_default_threshold))
    ]
    assert frame["threshold"].unique() == closest_threshold


def test_threshold_closest_match(pyplot, forest_binary_classification_with_train_test):
    """Check that the closest threshold is selected for data."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    # Create a threshold that is not in the list to test the closest match
    middle_index = len(display.thresholds) // 2
    threshold = (
        display.thresholds[middle_index] + display.thresholds[middle_index + 1]
    ) / 2 - 1e-6
    assert threshold not in display.thresholds

    display.plot(threshold_value=threshold)
    expected_title = (
        f"Confusion Matrix\nDecision threshold: {threshold:.2f}"
        + "\nData source: Test set"
    )
    assert display.figure_.get_suptitle() == expected_title

    np.testing.assert_allclose(
        display.ax_.collections[0].get_array(),
        display.frame(normalize=None, threshold_value=threshold)
        .pivot(index="true_label", columns="predicted_label", values="value")
        .reindex(index=display.display_labels, columns=display.display_labels)
        .values,
    )


def test_pos_label(pyplot, forest_binary_classification_with_train_test):
    """Check that the pos_label parameter works correctly."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]
    estimator.fit(X_train, y_train)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.metrics.confusion_matrix(pos_label="A")
    display.plot()
    assert display.ax_.get_xticklabels()[1].get_text() == "A*"
    assert display.ax_.get_yticklabels()[1].get_text() == "A*"

    display = report.metrics.confusion_matrix(pos_label="B")
    display.plot()
    assert display.ax_.get_xticklabels()[1].get_text() == "B*"
    assert display.ax_.get_yticklabels()[1].get_text() == "B*"


def test_plot_multiclass_no_threshold_in_title(
    pyplot, forest_multiclass_classification_with_train_test
):
    """Check that multiclass classification does not show threshold in title."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix()
    display.plot()

    expected_title = "Confusion Matrix" + "\nData source: Test set"
    assert display.figure_.get_suptitle() == expected_title
    assert "threshold" not in display.figure_.get_suptitle().lower()


def test_multiple_thresholdsdifferent_confusion_matrices(
    forest_binary_classification_with_train_test,
):
    """Check that different thresholds produce different confusion matrices."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    low_threshold = display.thresholds[len(display.thresholds) // 4]
    high_threshold = display.thresholds[3 * len(display.thresholds) // 4]

    cm = display.confusion_matrix
    cm_low = cm[cm["threshold"] == low_threshold]
    cm_high = cm[cm["threshold"] == high_threshold]

    assert cm_low["threshold"].iloc[0] != cm_high["threshold"].iloc[0]
    assert cm_low.shape == cm_high.shape


def test_threshold_values_are_sorted(forest_binary_classification_with_train_test):
    """Check that thresholds are sorted in ascending order."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    assert np.all(display.thresholds[:-1] <= display.thresholds[1:])


def test_threshold_values_are_unique(forest_binary_classification_with_train_test):
    """Check that thresholds contains unique values."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()

    assert len(display.thresholds) == len(np.unique(display.thresholds))


def test_threshold_greater_than_max(forest_binary_classification_with_train_test):
    """Check that a threshold greater than the maximum threshold is set to the maximum
    threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()
    frame = display.frame(threshold_value=1.1)
    assert frame["threshold"].unique() == display.thresholds[-1]
