import pandas as pd

from skore import ComparisonReport, MetricsSummaryDisplay


def test_default_is_wide(estimator_reports_binary_classification):
    """Default ``frame()`` returns wide layout for comparison-estimator reports."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])

    result = report.metrics.summarize().frame()

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.Index)
    assert result.columns.tolist() == ["DummyClassifier_1", "DummyClassifier_2"]


def test_data_source_both(estimator_reports_binary_classification):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame()

    assert result.index.to_list() == [
        "score",
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "log_loss",
        "brier_score",
        "fit_time",
        "predict_time",
    ]
    assert result.columns.to_list() == [
        "DummyClassifier_1 (train)",
        "DummyClassifier_1 (test)",
        "DummyClassifier_2 (train)",
        "DummyClassifier_2 (test)",
    ]


def test_format_wide(estimator_reports_binary_classification):
    """Compact format always returns a flat index and columns."""
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "score",
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "log_loss",
        "brier_score",
        "fit_time",
        "predict_time",
    ]
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_favorability(comparison_estimator_reports_binary_classification):
    """Check that the behaviour of `favorability` is correct."""
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.summarize()
    result = display.frame(favorability=True)
    assert set(result["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_has_estimator_columns(
    comparison_estimator_reports_binary_classification,
):
    """The wide frame exposes one column per compared estimator."""
    report = comparison_estimator_reports_binary_classification
    frame = report.metrics.summarize().frame(flat_index=False)

    assert isinstance(frame.index, pd.Index)
    assert frame.columns.tolist() == ["DummyClassifier_1", "DummyClassifier_2"]
    assert "split" not in frame.columns


def test_aggregate(comparison_estimator_reports_binary_classification):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    from pandas.testing import assert_frame_equal

    assert_frame_equal(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(),
    )
