import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import EstimatorReport


def test_multiple_thresholds_different_confusion_matrices(
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

    frame_low = display.frame(threshold_value=low_threshold)
    frame_high = display.frame(threshold_value=high_threshold)

    assert frame_low.shape == frame_high.shape
    assert frame_low["threshold"].iloc[0] != frame_high["threshold"].iloc[0]
    assert not np.array_equal(frame_low["value"].values, frame_high["value"].values)


@pytest.mark.parametrize("subplot_by", [None, "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "forest_binary_classification_with_train_test",
        "forest_multiclass_classification_with_train_test",
    ],
)
def test_subplot_by(pyplot, subplot_by, fixture_name, request):
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.confusion_matrix()
    if subplot_by == "invalid":
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: None or auto. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    else:
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_, mpl.axes.Axes)


@pytest.mark.parametrize(
    "estimator, expected_default_threshold",
    [(SVC(probability=False), 0), (LogisticRegression(), 0.5)],
)
def test_frame_default_threshold(
    binary_classification_train_test_split, estimator, expected_default_threshold
):
    """Check that frame() uses the right default threshold."""
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
    assert frame["threshold"].iloc[0] == closest_threshold


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
