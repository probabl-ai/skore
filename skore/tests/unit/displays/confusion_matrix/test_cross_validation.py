import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport


def test_multiple_thresholds_different_confusion_matrices(
    forest_binary_classification_data,
):
    """Check that different thresholds produce different confusion matrices."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

    low_threshold = display.thresholds[len(display.thresholds) // 4]
    high_threshold = display.thresholds[3 * len(display.thresholds) // 4]

    frame_low = display.frame(threshold_value=low_threshold)
    frame_high = display.frame(threshold_value=high_threshold)

    assert frame_low.shape == frame_high.shape
    assert frame_low["threshold"].iloc[0] != frame_high["threshold"].iloc[0]
    assert not np.array_equal(frame_low["value"].values, frame_high["value"].values)


def test_split_aggregation(pyplot, forest_binary_classification_data):
    """Check that confusion matrix values are aggregated across splits."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()
    fig = display.plot()
    ax = fig.axes[0]

    annotation_texts = [
        text.get_text()
        for text in ax.texts
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


@pytest.mark.parametrize("subplot_by", [None, "split", "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "forest_binary_classification_data",
        "forest_multiclass_classification_data",
    ],
)
def test_subplot_by(pyplot, subplot_by, fixture_name, request):
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X=X, y=y, splitter=3)
    display = report.metrics.confusion_matrix()
    if subplot_by == "invalid":
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: None, split or auto. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    elif subplot_by == "split":
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert len(axes) == 3
    else:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert isinstance(axes[0], mpl.axes.Axes)


def test_frame_default_returns_predict_based(forest_binary_classification_data):
    """Check that frame() returns the predict-based n x n matrix by default."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes * cv, 6)
    assert "threshold" not in frame.columns


def test_threshold_closest_match(forest_binary_classification_data):
    """Check that the closest threshold is selected for data."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

    middle_index = len(display.thresholds) // 2
    threshold = (
        display.thresholds[middle_index] + display.thresholds[middle_index + 1]
    ) / 2 - 1e-6
    assert threshold not in display.thresholds

    fig = display.plot(threshold_value=threshold)
    assert f"Decision threshold: {threshold:.2f}" in fig.get_suptitle()

    frame = display.frame(
        normalize=None, threshold_value=threshold, label=display.labels[-1]
    )
    aggregated = (
        frame.groupby(["true_label", "predicted_label"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    expected_values = aggregated.pivot(
        index="true_label", columns="predicted_label", values="mean"
    ).reindex(index=["0", "1"], columns=["0", "1"])

    ax = fig.axes[0]
    np.testing.assert_allclose(
        ax.collections[0].get_array(),
        expected_values.values,
    )


def test_pos_label(pyplot, forest_binary_classification_data):
    """Check that the report_pos_label parameter works correctly."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    estimator = LogisticRegression()
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv, pos_label="A")

    display = report.metrics.confusion_matrix()
    fig = display.plot()
    ax = fig.axes[0]
    assert ax.get_xticklabels()[1].get_text() == "A*"
    assert ax.get_yticklabels()[1].get_text() == "A*"

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv, pos_label="B")
    display = report.metrics.confusion_matrix()
    fig = display.plot()
    ax = fig.axes[0]
    assert ax.get_xticklabels()[1].get_text() == "B*"
    assert ax.get_yticklabels()[1].get_text() == "B*"
