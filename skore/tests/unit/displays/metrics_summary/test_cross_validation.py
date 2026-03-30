"""Tests for MetricsSummaryDisplay.frame() method with cross-validation DataFrames.

These tests focus on testing the display/formatting logic of MetricsSummaryDisplay
for cross-validation reports without depending on CrossValidationReport or summarize().
"""

import numpy as np
import pandas as pd
import pytest

from skore import CrossValidationReport


def test_summarize_classifier_without_predict_proba(
    custom_classifier_no_predict_proba_data,
):
    """Default metrics skip roc_auc, log_loss, and brier_score without predict_proba."""
    estimator, X, y = custom_classifier_no_predict_proba_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "Fit time (s)",
        "Predict time (s)",
    }

    result = display.frame(aggregate="mean", flat_index=True)
    assert result.shape == (7, 1)
    assert result.index.tolist() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "fit_time_s",
        "predict_time_s",
    ]
    assert result.columns.tolist() == ["customclassifierwithoutpredictproba_mean"]


def test_aggregate_mean(forest_binary_classification_data):
    """Test that aggregate='mean' returns only mean column."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate="mean")

    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == [("RandomForestClassifier", "mean")]
    assert result.shape == (10, 1)


def test_aggregate_mean_std(forest_binary_classification_data):
    """Test that aggregate=['mean', 'std'] returns both columns."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"])

    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == [
        ("RandomForestClassifier", "mean"),
        ("RandomForestClassifier", "std"),
    ]
    assert result.shape == (10, 2)


def test_aggregate_none(forest_binary_classification_data):
    """Test that aggregate=None returns individual splits."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=None)

    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.tolist() == [
        ("RandomForestClassifier", "Split #0"),
        ("RandomForestClassifier", "Split #1"),
    ]
    assert result.shape == (10, 2)


def test_favorability_with_aggregate_mean_std(forest_binary_classification_data):
    """
    Test that favorability column is correctly displayed with mean/std aggregation.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result_no_fav = display.frame(aggregate=["mean", "std"], favorability=False)
    assert result_no_fav.columns.tolist() == [
        ("RandomForestClassifier", "mean"),
        ("RandomForestClassifier", "std"),
    ]

    result_with_fav = display.frame(aggregate=["mean", "std"], favorability=True)
    assert result_with_fav.columns.tolist() == [
        ("RandomForestClassifier", "mean"),
        ("RandomForestClassifier", "std"),
        ("Favorability", ""),
    ]
    assert set(result_with_fav[("Favorability", "")]) == {"(↗︎)", "(↘︎)"}


def test_favorability_with_aggregate_none(forest_binary_classification_data):
    """Test that favorability column is correctly displayed with individual splits."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result_no_fav = display.frame(aggregate=None, favorability=False)
    assert result_no_fav.columns.tolist() == [
        ("RandomForestClassifier", "Split #0"),
        ("RandomForestClassifier", "Split #1"),
    ]

    result_with_fav = display.frame(aggregate=None, favorability=True)
    assert result_with_fav.columns.tolist() == [
        ("RandomForestClassifier", "Split #0"),
        ("RandomForestClassifier", "Split #1"),
        ("Favorability", ""),
    ]
    assert set(result_with_fav["Favorability", ""]) == {"(↗︎)", "(↘︎)"}


def test_flat_index_binary_classification(forest_binary_classification_data):
    """Test flat_index parameter with binary classification cross-validation data."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result_multi = display.frame(aggregate=["mean", "std"], flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Label / Average"]

    result_flat = display.frame(aggregate=["mean", "std"], flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    assert result_flat.index.tolist() == [
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


def test_multioutput_with_flat_index(linear_regression_multioutput_data):
    """Test flat_index with multioutput regression cross-validation data."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize(metric_kwargs={"multioutput": "raw_values"})

    result_multi = display.frame(aggregate=["mean", "std"], flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Output"]

    result_flat = display.frame(aggregate=["mean", "std"], flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    # Note: "R²" is lowercased
    assert result_flat.index.tolist() == [
        "r²_0",
        "r²_1",
        "rmse_0",
        "rmse_1",
        "fit_time_s",
        "predict_time_s",
    ]


def test_preserves_score_values_with_aggregate(forest_binary_classification_data):
    """Test that score values are correctly aggregated."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"])
    result_no_agg = display.frame(aggregate=None)
    accuracy_no_agg = result_no_agg.loc[("Accuracy", "")]

    assert result.loc[
        ("Accuracy", ""), ("RandomForestClassifier", "mean")
    ] == pytest.approx(np.mean(accuracy_no_agg))
    assert result.loc[
        ("Accuracy", ""), ("RandomForestClassifier", "std")
    ] == pytest.approx(np.std(accuracy_no_agg, ddof=1))


def test_flat_index_with_favorability(forest_binary_classification_data):
    """Test that flat_index and favorability work together for CV."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(
        aggregate=["mean", "std"], favorability=True, flat_index=True
    )
    assert result.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
        "favorability",
    ]

    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
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


def test_multiclass_classification(forest_multiclass_classification_data):
    """Test cross-validation with multiclass classification data."""
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"])

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Label / Average"]
    assert result.shape == (13, 2)


def test_with_mixed_favorability(forest_binary_classification_data):
    """Test CV with mixed favorability indicators."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"], favorability=True)

    assert ("Favorability", "") in result.columns
    assert result.index.names == ["Metric", "Label / Average"]
    assert result.loc[("Accuracy", ""), ("Favorability", "")] == "(↗︎)"
    assert result.loc[("Brier score", ""), ("Favorability", "")] == "(↘︎)"
