import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import CrossValidationReport


def test_normalization(pyplot, forest_binary_classification_data):
    """Check that normalized values are correctly computed."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()
    threshold_val = display.thresholds[len(display.thresholds) // 2]

    for split_idx in range(cv):
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


@pytest.mark.parametrize(
    "estimator, expected_default_threshold",
    [(SVC(probability=False), 0), (LogisticRegression(), 0.5)],
)
def test_frame_default_threshold(
    binary_classification_data, estimator, expected_default_threshold
):
    """Check that frame() uses the right default threshold."""
    X, y = binary_classification_data
    cv = 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

    frame = display.frame(threshold_value=None)
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    assert frame.shape == (n_classes * n_classes * cv, 7)

    for split_idx in range(cv):
        split_frame = frame.query(f"split == {split_idx}")
        assert split_frame["threshold"].nunique() == 1

        split_thresholds = display.confusion_matrix.query(f"split == {split_idx}")[
            "threshold"
        ].unique()
        closest_threshold = split_thresholds[
            np.argmin(abs(split_thresholds - expected_default_threshold))
        ]
        assert split_frame["threshold"].iloc[0] == closest_threshold


def test_threshold_closest_match(pyplot, forest_binary_classification_data):
    """Check that the closest threshold is selected for data."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()

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


def test_threshold_greater_than_max(forest_binary_classification_data):
    """Check that a threshold greater than the maximum threshold is set to the maximum
    threshold."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.confusion_matrix()
    frame = display.frame(threshold_value=1.1)

    for split_idx in range(cv):
        split_frame = frame.query(f"split == {split_idx}")
        assert split_frame["threshold"].nunique() == 1

        split_thresholds = display.confusion_matrix.query(f"split == {split_idx}")[
            "threshold"
        ].unique()
        split_max_threshold = np.max(split_thresholds)
        assert split_frame["threshold"].unique()[0] == split_max_threshold


def test_split_aggregation(pyplot, forest_binary_classification_data):
    """Check that confusion matrix values are aggregated across splits."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
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


@pytest.mark.parametrize("subplot_by", [None, "split", "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "forest_binary_classification_data",
        "forest_multiclass_classification_data",
    ],
)
def test_subplot_by(subplot_by, fixture_name, request):
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
        display.plot(subplot_by=subplot_by)
        assert len(display.ax_) == 3
    else:
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_, mpl.axes.Axes)
