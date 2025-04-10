"""Tests of metrics available in `ComparisonReport.metrics`."""

import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score
from skore import ComparisonReport, CrossValidationReport

expected_columns = pd.Index(["mean", "std"])


def comparison_report_classification():
    X, y = make_classification(class_sep=0.1, random_state=42)

    report = ComparisonReport(
        [
            CrossValidationReport(
                DummyClassifier(strategy="uniform", random_state=0), X, y
            ),
            CrossValidationReport(
                DummyClassifier(strategy="uniform", random_state=0), X, y, cv_splitter=3
            ),
        ]
    )

    return report


def comparison_report_regression():
    X, y = make_regression(random_state=42)

    report = ComparisonReport(
        [
            CrossValidationReport(DummyRegressor(), X, y),
            CrossValidationReport(DummyRegressor(), X, y, cv_splitter=3),
        ]
    )

    return report


def case_timings_no_predictions():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Fit time", "DummyClassifier_1"),
            ("Fit time", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "timings",
        expected_index,
        expected_columns,
    )


def case_timings_with_predictions():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Fit time", "DummyClassifier_1"),
            ("Predict time test", "DummyClassifier_1"),
            ("Predict time train", "DummyClassifier_1"),
            ("Fit time", "DummyClassifier_2"),
            ("Predict time test", "DummyClassifier_2"),
            ("Predict time train", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )

    report = comparison_report_classification()
    report.cache_predictions()
    return (
        report,
        "timings",
        expected_index,
        expected_columns,
    )


def case_accuracy():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Accuracy", "DummyClassifier_1"),
            ("Accuracy", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "accuracy",
        expected_index,
        expected_columns,
    )


def case_precision():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Precision", 0, "DummyClassifier_1"),
            ("Precision", 1, "DummyClassifier_1"),
            ("Precision", 0, "DummyClassifier_2"),
            ("Precision", 1, "DummyClassifier_2"),
        ],
        names=["Metric", "Label / Average", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "precision",
        expected_index,
        expected_columns,
    )


def case_recall():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Recall", 0, "DummyClassifier_1"),
            ("Recall", 1, "DummyClassifier_1"),
            ("Recall", 0, "DummyClassifier_2"),
            ("Recall", 1, "DummyClassifier_2"),
        ],
        names=["Metric", "Label / Average", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "recall",
        expected_index,
        expected_columns,
    )


def case_brier_score():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Brier score", "DummyClassifier_1"),
            ("Brier score", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "brier_score",
        expected_index,
        expected_columns,
    )


def case_roc_auc():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("ROC AUC", "DummyClassifier_1"),
            ("ROC AUC", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "roc_auc",
        expected_index,
        expected_columns,
    )


def case_log_loss():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Log loss", "DummyClassifier_1"),
            ("Log loss", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
        "log_loss",
        expected_index,
        expected_columns,
    )


def case_r2():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("R²", "DummyRegressor_1"),
            ("R²", "DummyRegressor_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_regression(),
        "r2",
        expected_index,
        expected_columns,
    )


def case_rmse():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("RMSE", "DummyRegressor_1"),
            ("RMSE", "DummyRegressor_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_regression(),
        "rmse",
        expected_index,
        expected_columns,
    )


@pytest.mark.parametrize(
    "case",
    [
        case_timings_no_predictions,
        case_timings_with_predictions,
        case_accuracy,
        case_precision,
        case_recall,
        case_brier_score,
        case_roc_auc,
        case_log_loss,
        case_r2,
        case_rmse,
    ],
)
def test_metrics(case):
    report, scoring, expected_index, expected_columns = case()

    result = getattr(report.metrics, scoring)()
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


def test_custom_metric():
    report, scoring, expected_index, expected_columns = case_accuracy()

    result = report.metrics.custom_metric(
        metric_function=accuracy_score,
        response_method="predict",
    )
    result = getattr(report.metrics, scoring)()
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)
