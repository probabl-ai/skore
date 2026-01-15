import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, get_scorer

from skore import ComparisonReport, MetricsSummaryDisplay


def test_data_source_external(
    binary_classification_data, comparison_estimator_reports_binary_classification
):
    """Check that `MetricsSummaryDisplay` works with an "X_y" data source."""
    X, y = binary_classification_data

    report = comparison_estimator_reports_binary_classification
    result = report.metrics.summarize(data_source="X_y", X=X[:10], y=y[:10]).frame()
    assert "Favorability" not in result.columns

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Accuracy", ""),
            ("Precision", 0),
            ("Precision", 1),
            ("Recall", 0),
            ("Recall", 1),
            ("ROC AUC", ""),
            ("Brier score", ""),
            ("Fit time (s)", ""),
            ("Predict time (s)", ""),
        ],
        names=["Metric", "Label / Average"],
    )
    expected_columns = pd.Index(
        ["DummyClassifier_1", "DummyClassifier_2"],
        name="Estimator",
    )

    pd.testing.assert_index_equal(result.index, expected_index)
    pd.testing.assert_index_equal(result.columns, expected_columns)

    assert len(report._cache) == 1
    cached_result = next(iter(report._cache.values()))
    pd.testing.assert_index_equal(cached_result.index, expected_index)
    pd.testing.assert_index_equal(cached_result.columns, expected_columns)


def test_data_source_both(
    binary_classification_data, comparison_estimator_reports_binary_classification
):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    X, y = binary_classification_data

    report = comparison_estimator_reports_binary_classification
    result = report.metrics.summarize(data_source="both", X=X[:10], y=y[:10]).frame()
    assert "Favorability" not in result.columns

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("Accuracy", ""),
            ("Precision", 0),
            ("Precision", 1),
            ("Recall", 0),
            ("Recall", 1),
            ("ROC AUC", ""),
            ("Brier score", ""),
            ("Fit time (s)", ""),
            ("Predict time (s)", ""),
        ],
        names=["Metric", "Label / Average"],
    )
    expected_columns = pd.Index(
        [
            "DummyClassifier_1 (train)",
            "DummyClassifier_1 (test)",
            "DummyClassifier_2 (train)",
            "DummyClassifier_2 (test)",
        ],
        name="Estimator",
    )

    pd.testing.assert_index_equal(result.index, expected_index)
    pd.testing.assert_index_equal(result.columns, expected_columns)

    assert len(report._cache) == 1
    cached_result = next(iter(report._cache.values()))
    pd.testing.assert_index_equal(cached_result.index, expected_index)
    pd.testing.assert_index_equal(cached_result.columns, expected_columns)


def test_flat_index(estimator_reports_binary_classification):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize(flat_index=True)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert result_df.shape == (9, 2)
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_indicator_favorability(
    comparison_cross_validation_reports_binary_classification,
):
    """Check that the behaviour of `indicator_favorability` is correct."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(indicator_favorability=True)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
    assert "Favorability" in result_df.columns
    indicator = result_df["Favorability"]
    assert indicator["Accuracy"].tolist() == ["(↗︎)"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]


def test_aggregate(
    comparison_estimator_reports_binary_classification,
):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    np.testing.assert_allclose(
        report.metrics.summarize(aggregate="mean").frame(),
        report.metrics.summarize().frame(),
    )


@pytest.mark.parametrize(
    "metric, metric_kwargs",
    [
        ("accuracy", None),
        ("neg_log_loss", None),
        (accuracy_score, {"response_method": "predict"}),
        (get_scorer("accuracy"), None),
    ],
)
def test_metric_single_list_equivalence(
    comparison_estimator_reports_binary_classification, metric, metric_kwargs
):
    """Check that passing a single string, callable, scorer is equivalent to passing a
    list with a single element."""
    report = comparison_estimator_reports_binary_classification
    result_single = report.metrics.summarize(
        metric=metric, metric_kwargs=metric_kwargs
    ).frame()
    result_list = report.metrics.summarize(
        metric=[metric], metric_kwargs=metric_kwargs
    ).frame()
    assert result_single.equals(result_list)
