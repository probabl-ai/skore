"""Tests of `ComparisonReport.metrics.summarize`."""

import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.dummy import DummyClassifier

from skore import ComparisonReport, CrossValidationReport


def test_aggregate_none(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(aggregate=None)

    assert result.columns.to_list() == ["Value"]
    assert result.index.names == ["Metric", "Label", "Estimator", "Split"]
    assert len(result) == 40


def test_aggregate_none_flat_index(
    comparison_cross_validation_reports_binary_classification,
):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None` and
    `flat_index=True`.
    """
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(aggregate=None, flat_index=True)

    assert result.columns.to_list() == ["Value"]
    assert len(result) == 40


def test_default(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with its default attributes."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame()

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyClassifier_1"),
                ("mean", "DummyClassifier_2"),
                ("std", "DummyClassifier_1"),
                ("std", "DummyClassifier_2"),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert len(result) == 10


def test_default_regression(comparison_cross_validation_reports_regression):
    """
    `MetricsSummaryDisplay` works as intended with its default attributes for regression
    models.
    """
    report = comparison_cross_validation_reports_regression
    result = report.metrics.summarize().frame()

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyRegressor_1"),
                ("mean", "DummyRegressor_2"),
                ("std", "DummyRegressor_1"),
                ("std", "DummyRegressor_2"),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert_index_equal(
        result.index,
        pd.Index(
            ["R²", "RMSE", "MAE", "MAPE", "Fit time (s)", "Predict time (s)"],
            name="Metric",
        ),
    )


def test_metric(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with the `metric` parameter."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(metric=["accuracy"]).frame(aggregate=None)

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert_index_equal(
        result.index,
        pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "DummyClassifier_1", "Split #0"),
                ("Accuracy", "DummyClassifier_1", "Split #1"),
                ("Accuracy", "DummyClassifier_2", "Split #0"),
                ("Accuracy", "DummyClassifier_2", "Split #1"),
            ],
            names=("Metric", "Estimator", "Split"),
        ),
    )


def test_favorability(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `favorability=True`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(favorability=True)

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyClassifier_1"),
                ("mean", "DummyClassifier_2"),
                ("std", "DummyClassifier_1"),
                ("std", "DummyClassifier_2"),
                ("Favorability", ""),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert len(result) == 10


def test_init_with_report_names(binary_classification_data):
    """
    If the estimators are passed as a dict, then the estimator names are the dict keys.
    """
    X, y = binary_classification_data

    report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y
    )
    report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=2), X=X, y=y
    )
    report = ComparisonReport({"model_1": report_1, "model_2": report_2})

    estimator_names = set(
        report.metrics.summarize().frame(aggregate=None).reset_index()["Estimator"]
    )
    assert estimator_names == {"model_1", "model_2"}


def test_cache_poisoning(binary_classification_data):
    """
    Computing metrics for a ComparisonReport should not influence the
    metrics computation for the internal CVReports.

    Non-regression test for https://github.com/probabl-ai/skore/issues/1706
    """
    X, y = binary_classification_data

    report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y
    )
    report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=2), X=X, y=y
    )
    report = ComparisonReport({"model_1": report_1, "model_2": report_2})
    report.metrics.summarize().frame(favorability=True)
    result = report_1.metrics.summarize().frame(aggregate=None, favorability=True)

    assert "Favorability" in result.columns


def test_aggregate_sequence_of_one_element(
    comparison_cross_validation_reports_binary_classification,
):
    """Passing a list of one string is the same as passing the string itself."""
    report = comparison_cross_validation_reports_binary_classification
    assert_frame_equal(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(aggregate=["mean"]),
    )
