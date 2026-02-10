import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import ComparisonReport, CrossValidationReport


@pytest.mark.parametrize("subplot_by", [None, "estimator", "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_cross_validation_reports_binary_classification",
        "comparison_cross_validation_reports_multiclass_classification",
    ],
)
def test_subplot_by(pyplot, subplot_by, fixture_name, request):
    """Check that the subplot_by parameter works correctly for comparison reports."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.confusion_matrix()
    if subplot_by in ["invalid", None]:
        err_msg = (
            "Invalid `subplot_by` parameter. Valid options are: estimator or auto. "
            f"Got '{subplot_by}' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by=subplot_by)
    elif subplot_by in ["estimator", "auto"]:
        display.plot(subplot_by=subplot_by)
        assert isinstance(display.ax_[0], mpl.axes.Axes)
        assert len(display.ax_) == len(report.reports_)


def test_split_aggregation(
    pyplot,
    comparison_cross_validation_reports_binary_classification,
    comparison_cross_validation_reports_binary_classification_figure_axes,
):
    """Check that confusion matrix values are aggregated across splits per estimator."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()
    _, axes = comparison_cross_validation_reports_binary_classification_figure_axes

    for ax in axes:
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


def test_estimator_names_in_confusion_matrix(
    comparison_cross_validation_reports_binary_classification,
):
    """Check that estimator names are correctly recorded in the confusion matrix."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.confusion_matrix()

    estimator_names = display.confusion_matrix["estimator"].unique()
    assert len(estimator_names) == 2
    assert set(estimator_names) == set(report.reports_.keys())


@pytest.mark.parametrize(
    "estimator, expected_default_threshold",
    [(SVC(probability=False), 0), (LogisticRegression(), 0.5)],
)
def test_frame_default_threshold(
    binary_classification_data, estimator, expected_default_threshold
):
    """Check that frame() uses the rightdefault threshold."""
    (X, y), cv = binary_classification_data, 3
    cv_report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    cv_report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report = ComparisonReport([cv_report_1, cv_report_2])
    display = report.metrics.confusion_matrix()

    frame = display.frame(threshold_value=None)
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    n_estimators = 2
    assert frame.shape == (n_classes * n_classes * n_estimators * cv, 7)

    for estimator_name in report.reports_:
        for split_idx in range(cv):
            estimator_split_frame = frame.query(
                f"estimator == '{estimator_name}'"
            ).query(f"split == {split_idx}")
            assert estimator_split_frame["threshold"].nunique() == 1
            estimator_split_thresholds = (
                display.confusion_matrix.query(f"estimator == '{estimator_name}'")
                .query(f"split == {split_idx}")["threshold"]
                .unique()
            )
            closest_threshold = estimator_split_thresholds[
                np.argmin(abs(estimator_split_thresholds - expected_default_threshold))
            ]
            assert estimator_split_frame["threshold"].iloc[0] == closest_threshold


def test_threshold_closest_match(pyplot, forest_binary_classification_data):
    """Check that the closest threshold is selected for data."""
    estimator, X, y = forest_binary_classification_data
    cv_report_1 = CrossValidationReport(estimator, X, y, splitter=3)
    cv_report_2 = CrossValidationReport(estimator, X, y, splitter=3)
    report = ComparisonReport([cv_report_1, cv_report_2])

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
    for idx, estimator in enumerate(report.reports_):
        frame = display.frame(normalize=None, threshold_value=threshold).query(
            f"estimator == '{estimator}'"
        )
        aggregated = (
            frame.groupby(["true_label", "predicted_label"])["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        expected_values = aggregated.pivot(
            index="true_label", columns="predicted_label", values="mean"
        ).reindex(index=display.display_labels, columns=display.display_labels)
        np.testing.assert_allclose(
            display.ax_[idx].collections[0].get_array(), expected_values.values
        )


def test_pos_label(pyplot, forest_binary_classification_data):
    """Check that the pos_label parameter works correctly."""
    estimator, X, y = forest_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y_labeled = labels[y]

    cv_report_1 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
    )
    cv_report_2 = CrossValidationReport(
        estimator,
        X=X,
        y=y_labeled,
        splitter=3,
    )
    report = ComparisonReport([cv_report_1, cv_report_2])

    display = report.metrics.confusion_matrix(pos_label="A")
    display.plot()
    for idx in range(len(report.reports_)):
        assert display.ax_[idx].get_xticklabels()[1].get_text() == "A*"
    # Only the first subplot has yticklabels
    assert display.ax_[0].get_yticklabels()[1].get_text() in "A*"

    display = report.metrics.confusion_matrix(pos_label="B")
    display.plot()
    for idx in range(len(report.reports_)):
        assert display.ax_[idx].get_xticklabels()[1].get_text() == "B*"
    assert display.ax_[0].get_yticklabels()[1].get_text() == "B*"
