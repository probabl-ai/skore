import numpy as np
import pandas as pd

from skore import ComparisonReport, MetricsSummaryDisplay


def test_data_source_both(
    binary_classification_data, estimator_reports_binary_classification
):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame()

    assert result.index.to_list() == [
        ("Accuracy", ""),
        ("Precision", "0"),
        ("Precision", "1"),
        ("Recall", "0"),
        ("Recall", "1"),
        ("ROC AUC", ""),
        ("Log loss", ""),
        ("Brier score", ""),
        ("Fit time (s)", ""),
        ("Predict time (s)", ""),
    ]
    assert result.columns.to_list() == [
        "DummyClassifier_1 (train)",
        "DummyClassifier_1 (test)",
        "DummyClassifier_2 (train)",
        "DummyClassifier_2 (test)",
    ]


def test_flat_index(estimator_reports_binary_classification):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame(flat_index=True)
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "log_loss",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_favorability(comparison_estimator_reports_binary_classification):
    """Check that the behaviour of `favorability` is correct."""
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.summarize()
    result = display.frame(favorability=True)
    assert set(result["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_aggregate(comparison_estimator_reports_binary_classification):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    np.testing.assert_allclose(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(),
    )
