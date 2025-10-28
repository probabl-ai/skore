import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from skore import ComparisonReport, EstimatorReport


@pytest.mark.parametrize("data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "accuracy",
            pd.DataFrame(
                [[0.6, 0.5]],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.Index(["Accuracy"], dtype="object", name="Metric"),
            ),
        ),
        (
            "precision",
            pd.DataFrame(
                [
                    [0.7777777777777778, 0.6666666666666666],
                    [0.45454545454545453, 0.36363636363636365],
                ],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Precision", 0), ("Precision", 1)],
                    names=["Metric", "Label / Average"],
                ),
            ),
        ),
        (
            "recall",
            pd.DataFrame(
                [
                    [0.5384615384615384, 0.46153846153846156],
                    [0.7142857142857143, 0.5714285714285714],
                ],
                columns=pd.Index(
                    ["DummyClassifier_1", "DummyClassifier_2"],
                    name="Estimator",
                ),
                index=pd.MultiIndex.from_tuples(
                    [("Recall", 0), ("Recall", 1)],
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
                index=pd.Index(["Brier score"], dtype="object", name="Metric"),
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
                index=pd.Index(["ROC AUC"], dtype="object", name="Metric"),
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
                index=pd.Index(["Log loss"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_binary_classification(
    metric_name,
    expected,
    data_source,
    comparison_estimator_reports_binary_classification,
):
    """Check the metrics work."""
    report = comparison_estimator_reports_binary_classification
    sub_report = next(iter(report.reports_.values()))
    X_test, y_test = sub_report.X_test, sub_report.y_test

    # ensure metric is valid
    if data_source == "X_y":
        result = getattr(report.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(report.metrics, metric_name)(data_source=data_source)
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    if data_source == "X_y":
        result = getattr(report.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(report.metrics, metric_name)(data_source=data_source)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (
            "rmse",
            pd.DataFrame(
                [[145.0950520567547, 145.0950520567547]],
                columns=pd.Index(
                    ["DummyRegressor_1", "DummyRegressor_2"],
                    name="Estimator",
                ),
                index=pd.Index(["RMSE"], dtype="object", name="Metric"),
            ),
        ),
        (
            "r2",
            pd.DataFrame(
                [[-0.0012562120106678343, -0.0012562120106678343]],
                columns=pd.Index(
                    ["DummyRegressor_1", "DummyRegressor_2"],
                    name="Estimator",
                ),
                index=pd.Index(["R²"], dtype="object", name="Metric"),
            ),
        ),
    ],
)
def test_regression(
    metric_name, expected, data_source, comparison_estimator_reports_regression
):
    """Check the metrics work."""
    comp = comparison_estimator_reports_regression
    sub_report = next(iter(comp.reports_.values()))
    X_test, y_test = sub_report.X_test, sub_report.y_test

    # ensure metric is valid
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    if data_source == "X_y":
        result = getattr(comp.metrics, metric_name)(
            data_source=data_source, X=X_test, y=y_test
        )
    else:
        result = getattr(comp.metrics, metric_name)()
    pd.testing.assert_frame_equal(result, expected)


def test_custom_metric_data_source_external(
    binary_classification_data, comparison_estimator_reports_binary_classification
):
    """Check that `custom_metric` works with an "X_y" data source."""
    X_test, y_test = binary_classification_data
    report = comparison_estimator_reports_binary_classification

    expected = pd.DataFrame(
        [[0.54, 0.39]],
        columns=pd.Index(["DummyClassifier_1", "DummyClassifier_2"], name="Estimator"),
        index=pd.Index(["Acc"], name="Metric"),
    )

    # ensure metric is valid
    result = report.metrics.custom_metric(
        metric_function=accuracy_score,
        response_method="predict",
        metric_name="Acc",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)

    # ensure metric is valid even from the cache
    result = report.metrics.custom_metric(
        metric_function=accuracy_score,
        response_method="predict",
        metric_name="Acc",
        data_source="X_y",
        X=X_test,
        y=y_test,
    )
    pd.testing.assert_frame_equal(result, expected)


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


def test_timings_flat_index(
    comparison_estimator_reports_binary_classification,
):
    """Check that time measurements have _s suffix with flat_index=True."""
    report = comparison_estimator_reports_binary_classification
    report.get_predictions(data_source="train")
    report.get_predictions(data_source="test")

    # Get metrics with flat_index=True
    results = report.metrics.summarize(flat_index=True).frame()

    # Check that expected time measurements are in index with _s suffix
    assert "fit_time_s" in results.index
    assert "predict_time_s" in results.index


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
    report = ComparisonReport([report_1, report_2])
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

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
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()


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
    result_both_labels = report.metrics.summarize(scoring=metric).frame().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_summarize_pos_label_overwrite(
    metric, logistic_binary_classification_with_train_test
):
    """Check that `pos_label` can be overwritten in `summarize`."""
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
        pos_label="B",
    )
    report_2 = EstimatorReport(
        clone(classifier),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pos_label="B",
    )
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = report.metrics.summarize(
        scoring=metric, pos_label=None
    ).frame()
    result = report.metrics.summarize(scoring=metric).frame().reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), report_name]
            == result_both_labels.loc[(metric.capitalize(), "B"), report_name]
        )

    result = (
        report.metrics.summarize(scoring=metric, pos_label="A").frame().reset_index()
    )
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), report_name]
            == result_both_labels.loc[(metric.capitalize(), "A"), report_name]
        )


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


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_overwrite(
    metric, logistic_binary_classification_with_train_test
):
    """Check that `pos_label` can be overwritten in `summarize`"""
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
    result_both_labels = getattr(report.metrics, metric)(pos_label=None)

    result = getattr(report.metrics, metric)(pos_label="B").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), report_name]
            == result_both_labels.loc[(metric.capitalize(), "B"), report_name]
        )

    result = getattr(report.metrics, metric)(pos_label="A").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), report_name]
            == result_both_labels.loc[(metric.capitalize(), "A"), report_name]
        )
