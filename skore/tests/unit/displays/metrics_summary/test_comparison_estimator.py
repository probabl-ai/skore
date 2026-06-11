import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor

from skore import ComparisonReport, MetricsSummaryDisplay, evaluate


def test_data_source_both(estimator_reports_binary_classification):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame()

    assert result.index.to_list() == [
        ("Score", ""),
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
        "score",
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


# Disambiguation of metrics with the same name


def test_score_disambiguated():
    """Estimators with different `.score` source code render as Score_1, Score_2."""
    X, y = make_regression(random_state=0)

    class A(DummyRegressor):
        def score(self, X, y):
            return 1

    class B(DummyRegressor):
        def score(self, X, y):
            return 2

    report = evaluate(
        [A(), B()],
        X,
        y,
        splitter=0.2,
    )

    result = report.metrics.summarize().frame()

    metric_names = result.index.tolist()
    assert metric_names[0] == "Score_1"
    assert metric_names[-1] == "Score_2"
    assert "Score" not in metric_names

    assert result.loc["Score_1", "A"] == 1
    assert result.loc["Score_2", "B"] == 2
    assert np.isnan(result.loc["Score_1", "B"])
    assert np.isnan(result.loc["Score_2", "A"])


def test_custom_metric_disambiguated(estimator_reports_regression):
    """Estimators with a custom metric with the same name but different source code
    render as Metric_1, Metric_2."""
    estimator_report_1, estimator_report_2 = estimator_reports_regression

    def metric(estimator, X, y):
        return 1

    estimator_report_1.metrics.add(metric)

    def metric(estimator, X, y):
        return 2

    estimator_report_2.metrics.add(metric)

    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize().frame()

    metric_names = result.index.tolist()
    assert metric_names[0] == "Metric_1"
    assert metric_names[-1] == "Metric_2"
    assert "Metric" not in metric_names

    assert result.loc["Metric_1", "DummyRegressor_1"] == 1
    assert result.loc["Metric_2", "DummyRegressor_2"] == 2
    assert np.isnan(result.loc["Metric_1", "DummyRegressor_2"])
    assert np.isnan(result.loc["Metric_2", "DummyRegressor_1"])
