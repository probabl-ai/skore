"""Tests of metrics available in `ComparisonReport.metrics`."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, CrossValidationReport

expected_columns = pd.MultiIndex.from_tuples(
    [
        ("mean", "DummyClassifier_1"),
        ("mean", "DummyClassifier_2"),
        ("std", "DummyClassifier_1"),
        ("std", "DummyClassifier_2"),
    ],
    names=[None, "Estimator"],
)


@pytest.fixture
def case_timings_no_predictions(
    comparison_cross_validation_reports_binary_classification,
):
    expected_index = pd.Index(["Fit time (s)"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "timings",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_timings_with_predictions(
    comparison_cross_validation_reports_binary_classification,
):
    expected_index = pd.Index(
        ["Fit time (s)", "Predict time test (s)", "Predict time train (s)"],
        name="Metric",
    )

    report = comparison_cross_validation_reports_binary_classification
    report.cache_predictions()
    return (
        report,
        "timings",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_accuracy(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Accuracy"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "accuracy",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_precision(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.MultiIndex.from_arrays(
        [
            ["Precision", "Precision"],
            pd.Index(["0", "1"], dtype="string", name="Label"),
        ],
        names=["Metric", "Label"],
    )
    return (
        comparison_cross_validation_reports_binary_classification,
        "precision",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_recall(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.MultiIndex.from_arrays(
        [
            ["Recall", "Recall"],
            pd.Index(["0", "1"], dtype="string", name="Label"),
        ],
        names=["Metric", "Label"],
    )
    return (
        comparison_cross_validation_reports_binary_classification,
        "recall",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_brier_score(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Brier score"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "brier_score",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_roc_auc(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["ROC AUC"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "roc_auc",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_log_loss(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Log loss"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "log_loss",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_r2(comparison_cross_validation_reports_regression):
    expected_index = pd.Index(["R²"], name="Metric")
    expected_columns = pd.MultiIndex.from_tuples(
        [
            ("mean", "DummyRegressor_1"),
            ("mean", "DummyRegressor_2"),
            ("std", "DummyRegressor_1"),
            ("std", "DummyRegressor_2"),
        ],
        names=[None, "Estimator"],
    )
    return (
        comparison_cross_validation_reports_regression,
        "r2",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_rmse(comparison_cross_validation_reports_regression):
    expected_index = pd.Index(["RMSE"], name="Metric")
    expected_columns = pd.MultiIndex.from_tuples(
        [
            ("mean", "DummyRegressor_1"),
            ("mean", "DummyRegressor_2"),
            ("std", "DummyRegressor_1"),
            ("std", "DummyRegressor_2"),
        ],
        names=[None, "Estimator"],
    )
    return (
        comparison_cross_validation_reports_regression,
        "rmse",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case(request):
    """Fixture to handle indirect parametrization of case fixtures."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "case",
    [
        "case_timings_no_predictions",
        "case_timings_with_predictions",
        "case_accuracy",
        "case_precision",
        "case_recall",
        "case_brier_score",
        "case_roc_auc",
        "case_log_loss",
        "case_r2",
        "case_rmse",
    ],
    indirect=True,
)
def test_metrics(case):
    report, metric, expected_index, expected_columns = case

    result = getattr(report.metrics, metric)()
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize(
    "case",
    [
        "case_timings_no_predictions",
        "case_timings_with_predictions",
        "case_accuracy",
        "case_precision",
        "case_recall",
        "case_brier_score",
        "case_roc_auc",
        "case_log_loss",
        "case_r2",
        "case_rmse",
    ],
    indirect=True,
)
def test_metrics_aggregate(case):
    """`aggregate` argument should be taken into account."""
    report, metric, expected_index, _ = case

    model = "DummyRegressor" if metric in ("r2", "rmse") else "DummyClassifier"
    expected_columns = pd.MultiIndex.from_tuples(
        [("mean", f"{model}_1"), ("mean", f"{model}_2")], names=[None, "Estimator"]
    )

    result = getattr(report.metrics, metric)(aggregate=["mean"])
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` is not set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y)
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y)
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label" not in fig.get_suptitle()

    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y, pos_label="A")
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y, pos_label="A")
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label: A" in fig.get_suptitle()


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `summarize`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result_both_labels["Label"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label"])


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `summarize`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = getattr(report.metrics, metric)().reset_index()
    assert result_both_labels["Label"].to_list() == ["A", "B"]
