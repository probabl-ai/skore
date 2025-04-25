import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import ConfusionMatrixDisplay


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(
        class_sep=0.1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


def test_confusion_matrix_display_binary_classification(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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


def test_confusion_matrix_display_multiclass_classification(
    pyplot,
    multiclass_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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
    assert len(display.display_labels) == n_classes

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (n_classes, n_classes)


def test_confusion_matrix_display_normalization(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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


def test_confusion_matrix_display_custom_labels(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    custom_labels = ["Negative", "Positive"]
    display = report.metrics.confusion_matrix(display_labels=custom_labels)

    assert display.display_labels == custom_labels
    frame = display.frame()
    assert list(frame.index) == custom_labels
    assert list(frame.columns) == custom_labels


def test_confusion_matrix_display_data_source(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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


def test_confusion_matrix_values_format(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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


def test_confusion_matrix_display_single_label(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.metrics.confusion_matrix(display_labels=["Only One Label"])
    assert isinstance(display, ConfusionMatrixDisplay)
    assert display.display_labels == ["Only One Label"]

    with pytest.raises(ValueError):
        display.plot()
