import matplotlib as mpl
import numpy as np
import pytest

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

    low_threshold = display.confusion_matrix_thresholded["threshold"].min()
    high_threshold = display.confusion_matrix_thresholded["threshold"].max()

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
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert isinstance(axes[0], mpl.axes.Axes)


def test_pos_label(pyplot, forest_binary_classification_with_train_test):
    """Check that the report_pos_label parameter works correctly."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]
    estimator.fit(X_train, y_train)
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label="A",
    )

    display = report.metrics.confusion_matrix()
    fig = display.plot()
    ax = fig.axes[0]
    assert ax.get_xticklabels()[1].get_text() == "A*"
    assert ax.get_yticklabels()[1].get_text() == "A*"

    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label="B",
    )
    display = report.metrics.confusion_matrix()
    fig = display.plot()
    ax = fig.axes[0]
    assert ax.get_xticklabels()[1].get_text() == "B*"
    assert ax.get_yticklabels()[1].get_text() == "B*"
