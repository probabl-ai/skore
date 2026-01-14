import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skore import ComparisonReport, EstimatorReport


def test_normalization(forest_binary_classification_with_train_test):
    """Check that normalized columns exist and have valid values."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )

    report_1 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comparison_report = ComparisonReport([report_1, report_2])
    display = comparison_report.metrics.confusion_matrix()
    threshold_val = display.thresholds[len(display.thresholds) // 2]

    for estimator_name in comparison_report.reports_:
        cm_true = display.frame(threshold_value=threshold_val, normalize="true").query(
            f"estimator == '{estimator_name}'"
        )
        pivoted_true = cm_true.pivot(
            index="true_label", columns="predicted_label", values="value"
        )
        np.testing.assert_allclose(pivoted_true.sum(axis=1), 1.0)

        cm_pred = display.frame(threshold_value=threshold_val, normalize="pred").query(
            f"estimator == '{estimator_name}'"
        )
        pivoted_pred = cm_pred.pivot(
            index="true_label", columns="predicted_label", values="value"
        )
        np.testing.assert_allclose(pivoted_pred.sum(axis=0), 1.0)

        cm_all = display.frame(threshold_value=threshold_val, normalize="all").query(
            f"estimator == '{estimator_name}'"
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
    binary_classification_train_test_split, estimator, expected_default_threshold
):
    """Check that frame() uses the right default threshold."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    comparison_report = ComparisonReport([report_1, report_2])
    display = comparison_report.metrics.confusion_matrix()

    frame = display.frame(threshold_value=None)
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.display_labels)
    n_estimators = 2
    assert frame.shape == (n_classes * n_classes * n_estimators, 7)

    for estimator_name in comparison_report.reports_:
        estimator_frame = frame.query(f"estimator == '{estimator_name}'")
        assert estimator_frame["threshold"].nunique() == 1
        estimator_thresholds = display.confusion_matrix.query(
            f"estimator == '{estimator_name}'"
        )["threshold"].unique()
        closest_threshold = estimator_thresholds[
            np.argmin(abs(estimator_thresholds - expected_default_threshold))
        ]
        assert estimator_frame["threshold"].iloc[0] == closest_threshold


def test_threshold_closest_match(pyplot, forest_binary_classification_with_train_test):
    """Check that the closest threshold is selected for data."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )

    report_1 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport([report_1, report_2])

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

    for idx, estimator in enumerate(report.reports_):
        np.testing.assert_allclose(
            display.ax_[idx].collections[0].get_array(),
            display.frame(normalize=None, threshold_value=threshold)
            .query(f"estimator == '{estimator}'")
            .pivot(index="true_label", columns="predicted_label", values="value")
            .reindex(index=display.display_labels, columns=display.display_labels)
            .values,
        )


def test_pos_label(pyplot, binary_classification_train_test_split):
    """Check that the pos_label parameter works correctly."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    labels = np.array(["A", "B"], dtype=object)
    y_train_labeled = labels[y_train]
    y_test_labeled = labels[y_test]

    report_1 = EstimatorReport(
        RandomForestClassifier(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="A",
    )
    report_2 = EstimatorReport(
        RandomForestClassifier(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="A",
    )
    comparison_report = ComparisonReport([report_1, report_2])

    display = comparison_report.metrics.confusion_matrix(pos_label="A")
    display.plot()
    for idx in range(len(comparison_report.reports_)):
        assert display.ax_[idx].get_xticklabels()[1].get_text() == "A*"
    # Only the first subplot has yticklabels
    assert display.ax_[0].get_yticklabels()[1].get_text() == "A*"

    display = comparison_report.metrics.confusion_matrix(pos_label="B")
    display.plot()
    for idx in range(len(comparison_report.reports_)):
        assert display.ax_[idx].get_xticklabels()[1].get_text() == "B*"
    assert display.ax_[0].get_yticklabels()[1].get_text() == "B*"


@pytest.mark.parametrize("subplot_by", [None, "estimator", "auto", "invalid"])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_estimator_reports_binary_classification",
        "comparison_estimator_reports_multiclass_classification",
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
