import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from skore import EstimatorReport
from skore._sklearn._plot import ConfusionMatrixDisplay


def test_binary_classification(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    """Check that the confusion matrix display is correct for binary classification."""
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
    assert display.ax_.get_title() == "Confusion Matrix"
    assert display.ax_.get_xlabel() == "Predicted label"
    assert display.ax_.get_ylabel() == "True label"

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, 2)


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

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes, n_classes)


def test_confusion_matrix(pyplot, forest_binary_classification_with_train_test):
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
    ]
    n_classes = len(display.display_labels)
    assert display.confusion_matrix.shape[0] == (n_classes * n_classes)


def test_normalization(pyplot, forest_binary_classification_with_train_test):
    """Check the behaviour of the `normalize` parameter for frame and plot"""
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

    display_true = report.metrics.confusion_matrix()
    display_true.plot(normalize="true")
    np.testing.assert_allclose(
        display_true.confusion_matrix.pivot(
            index="true_label", columns="predicted_label", values="normalized_by_true"
        ).sum(axis=1),
        1.0,
    )
    np.testing.assert_allclose(display_true.frame(normalize="true").sum(axis=1), 1.0)

    display_pred = report.metrics.confusion_matrix()
    display_pred.plot(normalize="pred")
    np.testing.assert_allclose(
        display_pred.confusion_matrix.pivot(
            index="true_label", columns="predicted_label", values="normalized_by_pred"
        ).sum(axis=0),
        1.0,
    )
    np.testing.assert_allclose(display_pred.frame(normalize="pred").sum(axis=0), 1.0)

    display_all = report.metrics.confusion_matrix()
    display_all.plot(normalize="all")
    assert display_all.confusion_matrix.pivot(
        index="true_label", columns="predicted_label", values="normalized_by_all"
    ).sum().sum() == pytest.approx(1.0)
    assert display_all.frame(normalize="all").sum().sum() == pytest.approx(1.0)


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

    assert display_test.confusion_matrix is not None  # Simple existence check
    assert not np.array_equal(  # Verify test vs train are different
        display_test.confusion_matrix,
        display_train.confusion_matrix,
    )
    assert np.array_equal(
        display_train.confusion_matrix,
        display_custom.confusion_matrix,
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
    assert len(display.figure_.axes) == 2


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
    assert len(display.ax_.texts) == 0

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"fmt": "d"})
    for text in display.ax_.texts:
        assert "." not in text.get_text() or text.get_text().endswith(".0")

    display = report.metrics.confusion_matrix()
    display.plot(heatmap_kwargs={"cbar": False})
    assert len(display.figure_.axes) == 1


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
    assert display.ax_.get_title() == "Confusion Matrix"

    n_classes = len(display.display_labels)
    assert len(display.ax_.get_xticks()) == n_classes
    assert len(display.ax_.get_yticks()) == n_classes

    xticklabels = [label.get_text() for label in display.ax_.get_xticklabels()]
    yticklabels = [label.get_text() for label in display.ax_.get_yticklabels()]
    assert xticklabels == [str(label) for label in display.display_labels]
    assert yticklabels == [str(label) for label in display.display_labels]


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
    assert frame.shape == (n_classes, n_classes)

    assert list(frame.index) == display.display_labels
    assert list(frame.columns) == display.display_labels

    expected = display.confusion_matrix.pivot(
        index="true_label", columns="predicted_label", values="count"
    )
    pd.testing.assert_frame_equal(frame, expected)


def test_not_implemented_error_for_non_estimator_report(
    pyplot, forest_binary_classification_with_train_test
):
    """Check that we raise NotImplementedError for non-estimator report types."""
    display = ConfusionMatrixDisplay(
        confusion_matrix=pd.DataFrame([[10, 5], [2, 20]]),
        display_labels=["0", "1"],
        report_type="cross-validation",
    )
    err_msg = (
        "`ConfusionMatrixDisplay` is only implemented for `EstimatorReport` for now."
    )
    with pytest.raises(NotImplementedError, match=err_msg):
        display.plot()


def test_threshold_display_creation(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that we can create a confusion matrix display with threshold support."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)

    assert isinstance(display, ConfusionMatrixDisplay)
    assert display.do_threshold is True
    assert display.thresholds_ is not None
    assert len(display.thresholds_) > 0
    assert "threshold" in display.confusion_matrix.columns


def test_threshold_display_without_threshold(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that do_threshold is False when threshold=False and that we raise an error
    when frame or plot is called with threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=False)

    assert display.do_threshold is False
    assert display.thresholds_ is None

    display = report.metrics.confusion_matrix(threshold=False)

    err_msg = (
        "threshold can only be used with binary classification and "
        "when `report.metrics.confusion_matrix\\(threshold=True\\)` is used."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.frame(threshold=0.5)

    with pytest.raises(ValueError, match=err_msg):
        display.plot(threshold=0.5)


def test_plot_with_threshold(pyplot, logistic_binary_classification_with_train_test):
    """Check that we can plot with a specific threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)

    display.plot(threshold=0.3)
    assert "threshold" in display.ax_.get_title().lower()


def test_plot_with_default_threshold(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that the default threshold (0.5) is used when not specified."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)
    display.plot()

    closest_threshold = display.thresholds_[
        np.argmin(np.abs(display.thresholds_ - 0.5))
    ]
    assert (
        display.ax_.get_title()
        == f"Confusion Matrix (threshold: {closest_threshold:.2f})"
    )


def test_frame_with_threshold(logistic_binary_classification_with_train_test):
    """Check that we can get a frame at a specific threshold."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)
    frame = display.frame(threshold=0.5)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, 2)


def test_frame_all_thresholds(logistic_binary_classification_with_train_test):
    """Check that we get all thresholds when threshold=None."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)
    frame = display.frame()

    assert isinstance(frame, pd.DataFrame)
    assert "threshold" in frame.columns
    assert len(frame) == len(display.thresholds_)


def test_threshold_normalization(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that normalization works with threshold support."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)

    display.plot(threshold=0.5, normalize="true")
    frame = display.frame(threshold=0.5, normalize="true")
    assert np.allclose(frame.sum(axis=1), np.ones(2))

    display.plot(threshold=0.5, normalize="pred")
    frame = display.frame(threshold=0.5, normalize="pred")
    assert np.allclose(frame.sum(axis=0), np.ones(2))

    display.plot(threshold=0.5, normalize="all")
    frame = display.frame(threshold=0.5, normalize="all")
    assert np.isclose(frame.sum().sum(), 1.0)


def test_threshold_closest_match(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that the closest threshold is selected."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)

    # Create a threshold that is not in the list to test the closest match
    middle_index = len(display.thresholds_) // 2
    threshold = (
        display.thresholds_[middle_index] + display.thresholds_[middle_index + 1]
    ) / 2 - 1e-6
    closest_threshold = display.thresholds_[middle_index]
    assert threshold not in display.thresholds_
    display.plot(threshold=threshold)
    assert (
        display.ax_.get_title()
        == f"Confusion Matrix (threshold: {closest_threshold:.2f})"
    )


def test_frame_plot_coincidence_with_threshold(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that the values in the frame and plot coincide when threshold is
    provided."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.confusion_matrix(threshold=True)
    frame = display.frame(threshold=0.5)
    frame_values = frame.values.flatten()
    display.plot(threshold=0.5)
    assert np.allclose(frame_values, display.ax_.collections[0].get_array().flatten())
