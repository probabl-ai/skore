"""Tests of metrics available in `CrossValidationComparisonReport.metrics`."""

import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from skore import CrossValidationComparisonReport, CrossValidationReport

expected_columns = pd.Index(["mean", "std"])


def comparison_report_classification():
    X, y = make_classification(class_sep=0.1, random_state=42)

    report = CrossValidationComparisonReport(
        [
            CrossValidationReport(DummyClassifier(), X, y),
            CrossValidationReport(DummyClassifier(), X, y, cv_splitter=3),
        ]
    )

    return report


def comparison_report_regression():
    X, y = make_regression(random_state=42)

    report = CrossValidationComparisonReport(
        [
            CrossValidationReport(DummyRegressor(), X, y),
            CrossValidationReport(DummyRegressor(), X, y, cv_splitter=3),
        ]
    )

    return report


def case_timings():
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Fit time", "DummyClassifier_1"),
            ("Predict time", "DummyClassifier_1"),
            ("Fit time", "DummyClassifier_2"),
            ("Predict time", "DummyClassifier_2"),
        ],
        names=["Metric", "Estimator"],
    )
    return (
        comparison_report_classification(),
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
        case_timings,
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
