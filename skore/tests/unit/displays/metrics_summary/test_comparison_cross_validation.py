"""Tests of `ComparisonReport.metrics.summarize`."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, get_scorer

from skore import ComparisonReport, CrossValidationReport, MetricsSummaryDisplay
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def test_aggregate_none(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(aggregate=None)
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()

    assert_index_equal(result_df.columns, pd.Index(["Value"]))
    assert result_df.index.names == ["Metric", "Label / Average", "Estimator", "Split"]
    assert len(result_df) == 90


def test_aggregate_none_flat_index(
    comparison_cross_validation_reports_binary_classification,
):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None` and
    `flat_index=True`.
    """
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(aggregate=None, flat_index=True).frame()

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert len(result) == 90


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
    assert len(result) == 9


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
        pd.Index(["R²", "RMSE", "Fit time (s)", "Predict time (s)"], name="Metric"),
    )


def test_aggregate_sequence_of_one_element(
    comparison_cross_validation_reports_binary_classification,
):
    """Passing a list of one string is the same as passing the string itself."""
    report = comparison_cross_validation_reports_binary_classification
    assert_frame_equal(
        report.metrics.summarize(aggregate="mean").frame(),
        report.metrics.summarize(aggregate=["mean"]).frame(),
    )


def test_aggregate_is_used_in_cache(
    comparison_cross_validation_reports_binary_classification,
):
    """`aggregate` should be used when computing the cache key.

    In other words, if you call `MetricsSummaryDisplay` twice with different values of
    `aggregate`, you should get a different result.
    """
    report = comparison_cross_validation_reports_binary_classification
    call1 = report.metrics.summarize(aggregate="mean").frame()
    call2 = report.metrics.summarize(aggregate=("mean", "std")).frame()
    assert list(call1.columns) != list(call2.columns)


def test_metric(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with the `metric` parameter."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(metric=["accuracy"], aggregate=None).frame()

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert_index_equal(
        result.index,
        pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "DummyClassifier_1", "Split #0"),
                ("Accuracy", "DummyClassifier_1", "Split #1"),
                ("Accuracy", "DummyClassifier_1", "Split #2"),
                ("Accuracy", "DummyClassifier_1", "Split #3"),
                ("Accuracy", "DummyClassifier_1", "Split #4"),
                ("Accuracy", "DummyClassifier_2", "Split #0"),
                ("Accuracy", "DummyClassifier_2", "Split #1"),
                ("Accuracy", "DummyClassifier_2", "Split #2"),
                ("Accuracy", "DummyClassifier_2", "Split #3"),
                ("Accuracy", "DummyClassifier_2", "Split #4"),
            ],
            names=("Metric", "Estimator", "Split"),
        ),
    )


def test_favorability(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `indicator_favorability=True`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(indicator_favorability=True).frame()

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
    assert len(result) == 9


def test_cache(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` results are cached."""
    report = comparison_cross_validation_reports_binary_classification
    with check_cache_changed(report._cache):
        result = report.metrics.summarize().frame()

    with check_cache_unchanged(report._cache):
        cached_result = report.metrics.summarize().frame()

    assert_frame_equal(result, cached_result)


def test_init_with_report_names(forest_binary_classification_data):
    """
    If the estimators are passed as a dict, then the estimator names are the dict keys.
    """

    estimator_1, X, y = forest_binary_classification_data
    estimator_2 = clone(estimator_1)
    cv_report1 = CrossValidationReport(estimator_1, X, y)
    cv_report2 = CrossValidationReport(estimator_2, X, y)

    comp = ComparisonReport({"r1": cv_report1, "r2": cv_report2})

    assert_index_equal(
        (
            comp.metrics.summarize(aggregate=None)
            .frame()
            .index.get_level_values("Estimator")
            .unique()
        ),
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_data_source_external(
    comparison_cross_validation_reports_binary_classification,
    binary_classification_data,
):
    """`MetricsSummaryDisplay` works as intended with `data_source="X_y"`."""
    report = comparison_cross_validation_reports_binary_classification
    X, y = binary_classification_data
    result = report.metrics.summarize(data_source="X_y", X=X, y=y).frame()

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
    assert len(result) == 9


def test_cache_poisoning(binary_classification_data):
    """Computing metrics for a ComparisonReport should not influence the
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
    report.metrics.summarize(indicator_favorability=True)
    result = report_1.metrics.summarize(
        aggregate=None, indicator_favorability=True
    ).frame()

    assert "Favorability" in result.columns


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
    comparison_cross_validation_reports_binary_classification, metric, metric_kwargs
):
    """Check that passing a single string, callable, scorer is equivalent to passing a
    list with a single element."""
    report = comparison_cross_validation_reports_binary_classification
    result_single = report.metrics.summarize(
        metric=metric, metric_kwargs=metric_kwargs
    ).frame()
    result_list = report.metrics.summarize(
        metric=[metric], metric_kwargs=metric_kwargs
    ).frame()
    assert result_single.equals(result_list)
