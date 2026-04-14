import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, EstimatorReport


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
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        assert isinstance(axes[0], mpl.axes.Axes)
        assert len(axes) == len(report.reports_)


def test_frame_default_returns_predict_based(binary_classification_train_test_split):
    """Check that frame() returns the predict-based n x n matrix by default."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    estimator = LogisticRegression()
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    comparison_report = ComparisonReport([report_1, report_2])
    display = comparison_report.metrics.confusion_matrix()

    frame = display.frame()
    assert isinstance(frame, pd.DataFrame)
    n_classes = len(display.labels)
    n_estimators = 2
    assert frame.shape == (n_classes * n_classes * n_estimators, 6)
    assert "threshold" not in frame.columns


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

    middle_index = len(display.thresholds) // 2
    threshold = (
        display.thresholds[middle_index] + display.thresholds[middle_index + 1]
    ) / 2 - 1e-6
    assert threshold not in display.thresholds

    fig = display.plot(threshold_value=threshold, label=display.labels[-1])
    axes = fig.axes
    assert f"Decision threshold: {threshold:.2f}" in fig.get_suptitle()

    for idx, estimator_name in enumerate(report.reports_):
        np.testing.assert_allclose(
            axes[idx].collections[0].get_array(),
            display.frame(
                normalize=None,
                threshold_value=threshold,
                label=display.labels[-1],
            )
            .query(f"estimator == '{estimator_name}'")
            .pivot(index="true_label", columns="predicted_label", values="value")
            .reindex(index=display.labels, columns=display.labels)
            .values,
        )


def test_pos_label(pyplot, binary_classification_train_test_split):
    """Check that the report_pos_label parameter works correctly."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    labels = np.array(["A", "B"], dtype=object)
    y_train_labeled = labels[y_train]
    y_test_labeled = labels[y_test]

    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="A",
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="A",
    )
    comparison_report = ComparisonReport([report_1, report_2])

    display = comparison_report.metrics.confusion_matrix()
    fig = display.plot()
    axes = fig.axes
    for idx in range(len(comparison_report.reports_)):
        assert axes[idx].get_xticklabels()[1].get_text() == "A*"
    # Only the first subplot has yticklabels
    assert axes[0].get_yticklabels()[1].get_text() == "A*"

    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="B",
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train_labeled,
        X_test=X_test,
        y_test=y_test_labeled,
        pos_label="B",
    )
    comparison_report = ComparisonReport([report_1, report_2])
    display = comparison_report.metrics.confusion_matrix()
    fig = display.plot()
    axes = fig.axes
    for idx in range(len(comparison_report.reports_)):
        assert axes[idx].get_xticklabels()[1].get_text() == "B*"
    assert axes[0].get_yticklabels()[1].get_text() == "B*"
