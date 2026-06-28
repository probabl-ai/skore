"""Tests of `ComparisonReport.metrics.summarize`."""

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier

from skore import ComparisonReport, CrossValidationReport


def test_format_auto_uses_long(
    comparison_cross_validation_reports_binary_classification,
):
    """Auto format uses long layout for comparison-cross-validation reports."""
    report = comparison_cross_validation_reports_binary_classification

    result = report.metrics.summarize().frame(format="auto")

    assert isinstance(result.index, pd.RangeIndex)
    assert "estimator" in result.columns


def test_aggregate_none(comparison_cross_validation_reports_binary_classification):
    """Compact format works with ``aggregate=None``."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(format="wide", aggregate=None)

    assert result.columns.to_list() == [
        "dummyclassifier_1_split_0",
        "dummyclassifier_1_split_1",
        "dummyclassifier_2_split_0",
        "dummyclassifier_2_split_1",
    ]
    assert len(result) == 11
    assert isinstance(result.index, pd.Index)


def test_default(comparison_cross_validation_reports_binary_classification):
    """Compact format works with default aggregation."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(format="wide")

    assert result.columns.tolist() == [
        "mean_dummyclassifier_1",
        "mean_dummyclassifier_2",
        "std_dummyclassifier_1",
        "std_dummyclassifier_2",
    ]
    assert len(result) == 11


def test_default_regression(comparison_cross_validation_reports_regression):
    """Compact format works for regression comparison reports."""
    report = comparison_cross_validation_reports_regression
    result = report.metrics.summarize().frame(format="wide")

    assert result.columns.tolist() == [
        "mean_dummyregressor_1",
        "mean_dummyregressor_2",
        "std_dummyregressor_1",
        "std_dummyregressor_2",
    ]
    assert result.index.tolist() == [
        "score",
        "r2",
        "rmse",
        "mae",
        "mape",
        "fit_time",
        "predict_time",
    ]


def test_aggregate_none_regression(comparison_cross_validation_reports_regression):
    """Compact format works with ``aggregate=None`` for regression."""
    report = comparison_cross_validation_reports_regression
    result = report.metrics.summarize().frame(format="wide", aggregate=None)

    assert isinstance(result.index, pd.Index)
    assert len(result) == 7
    assert len(result.columns) == 4


def test_metric(comparison_cross_validation_reports_binary_classification):
    """Compact format works with the ``metric`` parameter."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(metric=["accuracy"]).frame(
        format="wide", aggregate=None
    )

    assert result.columns.tolist() == [
        "dummyclassifier_1_split_0",
        "dummyclassifier_1_split_1",
        "dummyclassifier_2_split_0",
        "dummyclassifier_2_split_1",
    ]
    assert len(result) == 1


def test_favorability(comparison_cross_validation_reports_binary_classification):
    """Compact format works with ``favorability=True``."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(format="wide", favorability=True)

    assert result.columns.tolist() == [
        "mean_dummyclassifier_1",
        "mean_dummyclassifier_2",
        "std_dummyclassifier_1",
        "std_dummyclassifier_2",
        "favorability",
    ]
    assert len(result) == 11


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

    frame = report.metrics.summarize().frame(format="long", aggregate=None)
    assert set(frame["estimator"].unique()) == {"model_1", "model_2"}


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
    report.metrics.summarize().frame(format="wide", favorability=True)
    result = report_1.metrics.summarize().frame(
        format="wide", aggregate=None, favorability=True
    )

    assert "favorability" in result.columns


def test_aggregate_sequence_of_one_element(
    comparison_cross_validation_reports_binary_classification,
):
    """Passing a list of one string is the same as passing the string itself."""
    report = comparison_cross_validation_reports_binary_classification
    assert_frame_equal(
        report.metrics.summarize().frame(format="wide", aggregate="mean"),
        report.metrics.summarize().frame(format="wide", aggregate=["mean"]),
    )


def test_frame_has_estimator_and_split_columns(
    comparison_cross_validation_reports_binary_classification,
):
    """The tidy frame exposes both ``estimator`` and ``split`` columns."""
    report = comparison_cross_validation_reports_binary_classification
    frame = report.metrics.summarize().frame(format="long", aggregate=None)

    assert isinstance(frame.index, pd.RangeIndex)
    assert {"estimator", "split"}.issubset(frame.columns)
    assert frame["estimator"].nunique() == 2
    assert set(frame["split"]) == {0, 1}
