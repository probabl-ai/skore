"""Tests for ``MetricsSummaryDisplay.frame(format="wide")`` with cross-validation.

These tests focus on testing the display/formatting logic of MetricsSummaryDisplay
for cross-validation reports without depending on CrossValidationReport or summarize().
"""

import numpy as np
import pandas as pd
import pytest

from skore import CrossValidationReport


def test_aggregate_mean(forest_binary_classification_data):
    """Test that aggregate='mean' returns only mean column."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate="mean")

    assert isinstance(result, pd.Series)
    assert result.name == "randomforestclassifier_mean"
    assert isinstance(result.index, pd.Index)
    assert len(result) == 10


def test_aggregate_mean_std(forest_binary_classification_data):
    """Test that aggregate=['mean', 'std'] returns both columns."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"])

    assert isinstance(result.index, pd.Index)
    assert result.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
    ]
    assert result.shape == (10, 2)


def test_aggregate_none(forest_binary_classification_data):
    """Test that aggregate=None returns individual splits."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=None)

    assert isinstance(result.index, pd.Index)
    assert result.columns.tolist() == [
        "randomforestclassifier_split_0",
        "randomforestclassifier_split_1",
    ]
    assert result.shape == (10, 2)


def test_favorability_with_aggregate_mean_std(forest_binary_classification_data):
    """
    Test that favorability column is correctly displayed with mean/std aggregation.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result_no_fav = display.frame(
        format="wide", aggregate=["mean", "std"], favorability=False
    )
    assert result_no_fav.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
    ]

    result_with_fav = display.frame(
        format="wide", aggregate=["mean", "std"], favorability=True
    )
    assert result_with_fav.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
        "favorability",
    ]
    assert set(result_with_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_favorability_with_aggregate_none(forest_binary_classification_data):
    """Test that favorability column is correctly displayed with individual splits."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result_no_fav = display.frame(format="wide", aggregate=None, favorability=False)
    assert result_no_fav.columns.tolist() == [
        "randomforestclassifier_split_0",
        "randomforestclassifier_split_1",
    ]

    result_with_fav = display.frame(format="wide", aggregate=None, favorability=True)
    assert result_with_fav.columns.tolist() == [
        "randomforestclassifier_split_0",
        "randomforestclassifier_split_1",
        "favorability",
    ]
    assert set(result_with_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_format_wide_binary_classification(forest_binary_classification_data):
    """Compact format always returns a flat index for binary classification CV."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"])
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
        "fit_time",
        "predict_time",
    ]


def test_frame_with_multiindex_single_column(forest_binary_classification_data):
    """Single-column wide layout with MultiIndex columns returns a named Series."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(
        format="wide", aggregate="mean", with_multiindex=True, verbose_name=True
    )

    assert isinstance(result, pd.Series)
    assert result.name == "RandomForestClassifier_mean"


def test_frame_with_multiindex_cv(forest_binary_classification_data):
    """`with_multiindex=True` preserves column MultiIndex for CV mean/std layout."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(
        format="wide", aggregate=["mean", "std"], with_multiindex=True
    )

    assert isinstance(result.columns, pd.MultiIndex)


def test_format_wide_multioutput(linear_regression_multioutput_data):
    """Compact format returns a flat index for multioutput regression CV."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"])
    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
        "r2_0",
        "r2_1",
        "rmse_0",
        "rmse_1",
        "mae_0",
        "mae_1",
        "mape_0",
        "mape_1",
        "fit_time",
        "predict_time",
    ]


def test_preserves_score_values_with_aggregate(forest_binary_classification_data):
    """Test that score values are correctly aggregated."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"])
    result_no_agg = display.frame(format="wide", aggregate=None)
    accuracy_no_agg = result_no_agg.loc["accuracy"]

    assert result.loc["accuracy", "randomforestclassifier_mean"] == pytest.approx(
        np.mean(accuracy_no_agg)
    )
    assert result.loc["accuracy", "randomforestclassifier_std"] == pytest.approx(
        np.std(accuracy_no_agg, ddof=1)
    )


def test_format_wide_with_favorability(forest_binary_classification_data):
    """Compact format and favorability work together for CV."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"], favorability=True)
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
        "fit_time",
        "predict_time",
    ]


def test_data_source_both_favorability(forest_binary_classification_data):
    """Test favorability columns when data_source='both'."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    name = report.estimator_name_.lower()
    display = report.metrics.summarize(data_source="both")

    result = display.frame(format="wide", favorability=False)
    assert result.columns.tolist() == [
        f"{name}_(train)_mean",
        f"{name}_(train)_std",
        f"{name}_(test)_mean",
        f"{name}_(test)_std",
    ]

    result = display.frame(format="wide", favorability=True)
    assert result.columns.tolist() == [
        f"{name}_(train)_mean",
        f"{name}_(train)_std",
        f"{name}_(test)_mean",
        f"{name}_(test)_std",
        "favorability",
    ]


def test_data_source_both_format_wide(forest_binary_classification_data):
    """Compact format columns and index when data_source='both'."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    name = report.estimator_name_.lower()
    result = report.metrics.summarize(data_source="both").frame(format="wide")

    assert result.columns.tolist() == [
        f"{name}_(train)_mean",
        f"{name}_(train)_std",
        f"{name}_(test)_mean",
        f"{name}_(test)_std",
    ]
    assert result.index.tolist() == [
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


def test_multiclass_classification(forest_multiclass_classification_data):
    """Test cross-validation with multiclass classification data."""
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"])

    assert isinstance(result.index, pd.Index)
    assert result.shape == (16, 2)


def test_with_mixed_favorability(forest_binary_classification_data):
    """Test CV with mixed favorability indicators."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="wide", aggregate=["mean", "std"], favorability=True)

    assert "favorability" in result.columns
    assert isinstance(result.index, pd.Index)
    assert result.loc["accuracy", "favorability"] == "(↗︎)"
    assert result.loc["brier_score", "favorability"] == "(↘︎)"


def test_frame_has_split_column(forest_binary_classification_data):
    """The tidy frame exposes one row per split via a ``split`` column."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    frame = report.metrics.summarize().frame(format="long", aggregate=None)

    assert isinstance(frame.index, pd.RangeIndex)
    assert "split" in frame.columns
    assert "estimator" not in frame.columns
    assert set(frame["split"]) == {0, 1}


def test_long_frame_aggregate_mean_std(forest_binary_classification_data):
    """Long format with aggregate exposes mean/std rows, not per-split rows."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    frame = display.frame(format="long", aggregate=["mean", "std"])
    wide = display.frame(format="wide", aggregate=["mean", "std"])

    assert isinstance(frame.index, pd.RangeIndex)
    assert "aggregate" in frame.columns
    assert "split" not in frame.columns
    assert set(frame["aggregate"]) == {"mean", "std"}

    accuracy_mean = frame.loc[
        (frame["metric"] == "accuracy") & (frame["aggregate"] == "mean"), "value"
    ].iloc[0]
    assert accuracy_mean == pytest.approx(
        wide.loc["accuracy", "randomforestclassifier_mean"]
    )


def test_format_auto_uses_wide_for_cv(forest_binary_classification_data):
    """Auto format uses wide layout for single cross-validation reports."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(format="auto", aggregate=["mean", "std"])

    assert isinstance(result.index, pd.Index)
    assert result.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
    ]


def test_repr_frame_uses_long_for_comparison(
    forest_binary_classification_data,
):
    """Internal repr uses long format for comparison reports."""
    from skore import ComparisonReport

    estimator, X, y = forest_binary_classification_data
    reports = {
        f"est_{i}": CrossValidationReport(estimator, X=X, y=y, splitter=2)
        for i in range(2)
    }
    display = ComparisonReport(reports).metrics.summarize()

    result = display._repr_frame()

    assert isinstance(result.index, pd.RangeIndex)
    assert "aggregate" in result.columns
    assert "estimator" in result.columns
