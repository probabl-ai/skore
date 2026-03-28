import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from skore import ComparisonReport, EstimatorReport


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "accuracy",
            pd.DataFrame(
                [[0.45, 0.55]],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Accuracy"], name="Metric"),
            ),
        ),
        (
            "precision",
            pd.DataFrame(
                [
                    [0.333333, 0.444444],
                    [0.545455, 0.636364],
                ],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Precision", "0"), ("Precision", "1")],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "recall",
            pd.DataFrame(
                [
                    [0.375, 0.5],
                    [0.5, 0.583333],
                ],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Recall", "0"), ("Recall", "1")],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "brier_score",
            pd.DataFrame(
                [[0.25, 0.25]],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Brier score"], name="Metric"),
            ),
        ),
        (
            "roc_auc",
            pd.DataFrame(
                [[0.5, 0.5]],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.Index(["ROC AUC"], name="Metric"),
            ),
        ),
        (
            "log_loss",
            pd.DataFrame(
                [[0.693147, 0.693147]],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Log loss"], name="Metric"),
            ),
        ),
    ],
)
def test_binary_classification(
    metric_name, expected, comparison_estimator_reports_binary_classification
):
    """Check the metrics work."""
    report = comparison_estimator_reports_binary_classification

    # ensure metric is valid
    result = getattr(report.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    result = getattr(report.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "rmse",
            pd.DataFrame(
                [[94.426101, 94.426101]],
                columns=pd.Index(
                    ["DummyRegressor_1", "DummyRegressor_2"],
                    name="Estimator",
                ),
                index=pd.Index(["RMSE"], name="Metric"),
            ),
        ),
        (
            "r2",
            pd.DataFrame(
                [[-0.061173, -0.061173]],
                columns=pd.Index(
                    ["DummyRegressor_1", "DummyRegressor_2"],
                    name="Estimator",
                ),
                index=pd.Index(["R²"], name="Metric"),
            ),
        ),
    ],
)
def test_regression(metric_name, expected, comparison_estimator_reports_regression):
    """Check the metrics work."""
    comp = comparison_estimator_reports_regression
    # ensure metric is valid
    result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)

    # ensure metric is valid even from the cache
    result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)


def test_timings(comparison_estimator_reports_binary_classification):
    """Check the general behaviour of the `timings` method."""
    report = comparison_estimator_reports_binary_classification
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)"]
    assert timings.columns.tolist() == list(report.reports_.keys())

    report.get_predictions(data_source="train")
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == ["Fit time (s)", "Predict time train (s)"]
    assert timings.columns.tolist() == list(report.reports_.keys())

    report.get_predictions(data_source="test")
    timings = report.metrics.timings()
    assert isinstance(timings, pd.DataFrame)
    assert timings.index.tolist() == [
        "Fit time (s)",
        "Predict time train (s)",
        "Predict time test (s)",
    ]
    assert timings.columns.tolist() == list(report.reports_.keys())


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_pos_label(
    pyplot, metric, logistic_binary_classification_with_train_test
):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    classifier, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]
    report_1 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # TODO: test when pos_label is not set

    report_1 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pos_label="A",
    )
    report_2 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pos_label="A",
    )
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label: A" in fig.get_suptitle()


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_summarize_pos_label_default(
    metric, logistic_binary_classification_with_train_test
):
    """Check the default behaviour of `pos_label` in `summarize`."""
    classifier, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]

    report_1 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_default(
    metric, logistic_binary_classification_with_train_test
):
    """Check the default behaviour of `pos_label` in `summarize`."""
    classifier, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]
    report_1 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = getattr(report.metrics, metric)().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])
