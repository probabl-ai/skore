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
    assert hasattr(display, "text_")
    assert display.confusion_matrix.shape == (2, 2)
    assert len(display.display_labels) == 2
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
    assert display.confusion_matrix.shape == (n_classes, n_classes)
    display.plot()
    assert len(display.display_labels) == n_classes

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes, n_classes)


def test_normalization(pyplot, forest_binary_classification_with_train_test):
    """Check the behaviour of the `normalize` parameter."""
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

    display_true = report.metrics.confusion_matrix(normalize="true")
    display_true.plot()
    assert np.allclose(display_true.confusion_matrix.sum(axis=1), np.ones(2))

    display_pred = report.metrics.confusion_matrix(normalize="pred")
    display_pred.plot()
    assert np.allclose(display_pred.confusion_matrix.sum(axis=0), np.ones(2))

    display_all = report.metrics.confusion_matrix(normalize="all")
    display_all.plot()
    assert np.isclose(display_all.confusion_matrix.sum(), 1.0)


def test_custom_labels(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    """Check the behaviour when passing custom labels to the confusion matrix display.
    """
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    custom_labels = ["Negative", "Positive"]
    display = report.metrics.confusion_matrix()
    display.plot(display_labels=custom_labels)

    assert display.display_labels == custom_labels
    frame = display.frame()
    assert list(frame.index) == custom_labels
    assert list(frame.columns) == custom_labels


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


def test_values_format(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    """Check that we can customize the format of the confusion matrix values."""
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display_int = report.metrics.confusion_matrix(values_format="d")
    display_float = report.metrics.confusion_matrix(values_format=".2f")
    assert np.array_equal(
        display_int.confusion_matrix,
        display_float.confusion_matrix,
    )


def test_single_label(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    """Check that we can pass a single label to the confusion matrix display."""
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.metrics.confusion_matrix()
    assert isinstance(display, ConfusionMatrixDisplay)

    with pytest.raises(ValueError):
        display.plot(display_labels=["Only One Label"])
