import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationComparisonReport, CrossValidationReport


def case_different_split_numbers():
    X, y = make_classification(random_state=42)

    report = CrossValidationComparisonReport(
        [
            CrossValidationReport(LogisticRegression(), X, y),
            CrossValidationReport(LogisticRegression(), X, y, cv_splitter=3),
        ]
    )

    kwargs = {"aggregate": None}

    expected_columns = pd.Index(["m1", "m2"], name="Estimator")
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Precision", 0, "Split #0"),
            ("Precision", 0, "Split #1"),
            ("Precision", 0, "Split #2"),
            ("Precision", 0, "Split #3"),
            ("Precision", 0, "Split #4"),
            ("Precision", 1, "Split #0"),
            ("Precision", 1, "Split #1"),
            ("Precision", 1, "Split #2"),
            ("Precision", 1, "Split #3"),
            ("Precision", 1, "Split #4"),
            ("Recall", 0, "Split #0"),
            ("Recall", 0, "Split #1"),
            ("Recall", 0, "Split #2"),
            ("Recall", 0, "Split #3"),
            ("Recall", 0, "Split #4"),
            ("Recall", 1, "Split #0"),
            ("Recall", 1, "Split #1"),
            ("Recall", 1, "Split #2"),
            ("Recall", 1, "Split #3"),
            ("Recall", 1, "Split #4"),
            ("ROC AUC", "", "Split #0"),
            ("ROC AUC", "", "Split #1"),
            ("ROC AUC", "", "Split #2"),
            ("ROC AUC", "", "Split #3"),
            ("ROC AUC", "", "Split #4"),
            ("Brier score", "", "Split #0"),
            ("Brier score", "", "Split #1"),
            ("Brier score", "", "Split #2"),
            ("Brier score", "", "Split #3"),
            ("Brier score", "", "Split #4"),
            ("Fit time", "", "Split #0"),
            ("Fit time", "", "Split #1"),
            ("Fit time", "", "Split #2"),
            ("Fit time", "", "Split #3"),
            ("Fit time", "", "Split #4"),
            ("Predict time", "", "Split #0"),
            ("Predict time", "", "Split #1"),
            ("Predict time", "", "Split #2"),
            ("Predict time", "", "Split #3"),
            ("Predict time", "", "Split #4"),
        ],
        names=["Metric", "Label / Average", "Split"],
    )

    return report, kwargs, expected_index, expected_columns


def case_flat_index_different_split_numbers():
    report, kwargs, expected_index, expected_columns = case_different_split_numbers()

    kwargs = {"aggregate": None, "flat_index": True}

    expected_columns = pd.Index(["m1", "m2"], name="Estimator")
    expected_index = pd.Index(
        [
            "precision_0_split_0",
            "precision_0_split_1",
            "precision_0_split_2",
            "precision_0_split_3",
            "precision_0_split_4",
            "precision_1_split_0",
            "precision_1_split_1",
            "precision_1_split_2",
            "precision_1_split_3",
            "precision_1_split_4",
            "recall_0_split_0",
            "recall_0_split_1",
            "recall_0_split_2",
            "recall_0_split_3",
            "recall_0_split_4",
            "recall_1_split_0",
            "recall_1_split_1",
            "recall_1_split_2",
            "recall_1_split_3",
            "recall_1_split_4",
            "roc_auc_split_0",
            "roc_auc_split_1",
            "roc_auc_split_2",
            "roc_auc_split_3",
            "roc_auc_split_4",
            "brier_score_split_0",
            "brier_score_split_1",
            "brier_score_split_2",
            "brier_score_split_3",
            "brier_score_split_4",
            "fit_time_split_0",
            "fit_time_split_1",
            "fit_time_split_2",
            "fit_time_split_3",
            "fit_time_split_4",
            "predict_time_split_0",
            "predict_time_split_1",
            "predict_time_split_2",
            "predict_time_split_3",
            "predict_time_split_4",
        ]
    )

    return report, kwargs, expected_index, expected_columns


def case_aggregate_different_split_numbers():
    report, _, _, _ = case_different_split_numbers()

    kwargs = {}

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Precision", 0, "m1"),
            ("Precision", 1, "m1"),
            ("Recall", 0, "m1"),
            ("Recall", 1, "m1"),
            ("ROC AUC", "", "m1"),
            ("Brier score", "", "m1"),
            ("Fit time", "", "m1"),
            ("Predict time", "", "m1"),
            ("Precision", 0, "m2"),
            ("Precision", 1, "m2"),
            ("Recall", 0, "m2"),
            ("Recall", 1, "m2"),
            ("ROC AUC", "", "m2"),
            ("Brier score", "", "m2"),
            ("Fit time", "", "m2"),
            ("Predict time", "", "m2"),
        ],
        names=["Metric", "Label / Average", "Estimator"],
    )

    expected_columns = pd.Index(["mean", "std"])

    return report, kwargs, expected_index, expected_columns


@pytest.mark.parametrize(
    "case",
    [
        case_different_split_numbers,
        case_flat_index_different_split_numbers,
        case_aggregate_different_split_numbers,
    ],
)
def test_report_metrics(case):
    report, kwargs, expected_index, expected_columns = case()

    result = report.metrics.report_metrics(**kwargs)
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)
