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
    display = report.metrics.confusion_matrix(display_labels=custom_labels)

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

    display_int = report.metrics.confusion_matrix()
    display_int.plot(text_kwargs={"values_format": "d"})
    display_float = report.metrics.confusion_matrix()
    display_float.plot(text_kwargs={"values_format": ".2f"})
    assert np.array_equal(
        display_int.confusion_matrix,
        display_float.confusion_matrix,
    )


def test_single_label(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    """Check that we can't pass a single label to the confusion matrix display."""
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    err_msg = "display_labels must have length equal to number of classes"
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.confusion_matrix(display_labels=["Only One Label"])


def test_imshow_kwargs(pyplot, forest_binary_classification_with_train_test):
    """Check that we can pass keyword arguments to the imshow call."""
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
    display.plot(imshow_kwargs={"alpha": 0.5, "vmin": 0, "vmax": 100})

    assert len(display.ax_.images) == 1
    imshow = display.ax_.images[0]
    assert imshow.get_alpha() == 0.5
    assert imshow.get_clim() == (0, 100)


def test_set_style(pyplot, forest_binary_classification_with_train_test):
    """Check that the set_style method works with imshow_kwargs."""
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

    display.set_style(imshow_kwargs={"alpha": 0.5})
    display.plot()

    assert display.ax_.images[0].get_alpha() == 0.5

    display.plot(imshow_kwargs={"alpha": 0.8})
    assert display.ax_.images[0].get_alpha() == 0.8


def test_colorbar(pyplot, forest_binary_classification_with_train_test):
    """Check that we can control the colorbar."""
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
    display.plot(colorbar=True)
    assert len(display.figure_.axes) == 2

    display = report.metrics.confusion_matrix()
    display.plot(colorbar=False)
    assert len(display.figure_.axes) == 1


def test_include_values(pyplot, forest_binary_classification_with_train_test):
    """Check that we can control whether values are displayed in cells."""
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
    display.plot(include_values=True)
    assert display.text_ is not None
    assert isinstance(display.text_, np.ndarray)
    assert display.text_.size > 0
    assert all(isinstance(text, mpl.text.Text) for text in display.text_.flat)

    display = report.metrics.confusion_matrix()
    display.plot(include_values=False)
    assert display.text_ is None


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
    assert xticklabels == display.display_labels
    assert yticklabels == display.display_labels


def test_cmap(pyplot, forest_binary_classification_with_train_test):
    """Check that we can change the colormap."""
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
    assert display.ax_.images[0].get_cmap().name == "Blues"

    display.plot(cmap="Reds")
    assert display.ax_.images[0].get_cmap().name == "Reds"


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

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == display.confusion_matrix.shape

    assert list(frame.index) == display.display_labels
    assert list(frame.columns) == display.display_labels

    np.testing.assert_array_equal(frame.values, display.confusion_matrix)


def test_frame_with_normalization(forest_binary_classification_with_train_test):
    """Check that the frame method works correctly with normalization."""
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
    frame_true = display_true.frame()
    row_sums = frame_true.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(row_sums)))

    display_pred = report.metrics.confusion_matrix(normalize="pred")
    frame_pred = display_pred.frame()
    col_sums = frame_pred.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(len(col_sums)))

    display_all = report.metrics.confusion_matrix(normalize="all")
    frame_all = display_all.frame()
    total_sum = frame_all.sum().sum()
    np.testing.assert_allclose(total_sum, 1.0)


def test_text_formatting(pyplot, forest_binary_classification_with_train_test):
    """Check that text values are formatted correctly."""
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
    display.plot(text_kwargs={"values_format": "d"})

    for text in display.text_.flat:
        if text is not None:
            assert isinstance(text, mpl.text.Text)
            assert "." not in text.get_text() or text.get_text().endswith(".0")

    display = report.metrics.confusion_matrix(normalize="true")
    display.plot(text_kwargs={"values_format": ".3f"})

    for text in display.text_.flat:
        if text is not None:
            assert isinstance(text, mpl.text.Text)
            assert "." in text.get_text()


def test_not_implemented_error_for_non_estimator_report(
    pyplot, forest_binary_classification_with_train_test
):
    """Check that we raise NotImplementedError for non-estimator report types."""
    cm = np.array([[10, 5], [2, 20]])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["0", "1"],
        normalize=None,
        report_type="cross-validation",
    )

    err_msg = (
        "`ConfusionMatrixDisplay` is only implemented for `EstimatorReport` for now."
    )
    with pytest.raises(NotImplementedError, match=err_msg):
        display.plot()
