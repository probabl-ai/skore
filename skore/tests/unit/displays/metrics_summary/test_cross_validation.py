"""Tests for ``MetricsSummaryDisplay.frame()`` with cross-validation.

These tests focus on testing the display/formatting logic of MetricsSummaryDisplay
for cross-validation reports without depending on CrossValidationReport or summarize().
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from skore import CrossValidationReport


def test_aggregate_mean(forest_binary_classification_data):
    """Test that aggregate='mean' returns only mean column."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate="mean")

    assert isinstance(result, pd.Series)
    assert result.name == "randomforestclassifier_mean"
    assert isinstance(result.index, pd.Index)
    assert len(result) == 10


def test_aggregate_mean_std(forest_binary_classification_data):
    """Test that aggregate=['mean', 'std'] returns both columns."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"])

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

    result = display.frame(aggregate=None)

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

    result_no_fav = display.frame(aggregate=["mean", "std"], favorability=False)
    assert result_no_fav.columns.tolist() == [
        "randomforestclassifier_mean",
        "randomforestclassifier_std",
    ]

    result_with_fav = display.frame(aggregate=["mean", "std"], favorability=True)
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

    result_no_fav = display.frame(aggregate=None, favorability=False)
    assert result_no_fav.columns.tolist() == [
        "randomforestclassifier_split_0",
        "randomforestclassifier_split_1",
    ]

    result_with_fav = display.frame(aggregate=None, favorability=True)
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

    result = display.frame(aggregate=["mean", "std"])
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

    result = display.frame(aggregate="mean", flat_index=False, verbose_name=True)

    assert isinstance(result, pd.Series)
    assert result.name == "RandomForestClassifier_mean"


def test_frame_with_multiindex_cv(forest_binary_classification_data):
    """`flat_index=False` preserves column MultiIndex for CV mean/std layout."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"], flat_index=False)

    assert isinstance(result.columns, pd.MultiIndex)


def test_format_wide_multioutput(linear_regression_multioutput_data):
    """Compact format returns a flat index for multioutput regression CV."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"])
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

    result = display.frame(aggregate=["mean", "std"])
    result_no_agg = display.frame(aggregate=None)
    accuracy_no_agg = result_no_agg.loc["accuracy"]

    assert result.loc["accuracy", "randomforestclassifier_mean"] == pytest.approx(
        np.mean(accuracy_no_agg)
    )
    assert result.loc["accuracy", "randomforestclassifier_std"] == pytest.approx(
        np.std(accuracy_no_agg, ddof=1)
    )


def test_data_source_both_favorability(forest_binary_classification_data):
    """Test favorability columns when data_source='both'."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    name = report.estimator_name_.lower()
    display = report.metrics.summarize(data_source="both")

    result = display.frame(favorability=False)
    assert result.columns.tolist() == [
        f"{name}_(train)_mean",
        f"{name}_(train)_std",
        f"{name}_(test)_mean",
        f"{name}_(test)_std",
    ]

    result = display.frame(favorability=True)
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
    result = report.metrics.summarize(data_source="both").frame()

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

    result = display.frame(aggregate=["mean", "std"])

    assert isinstance(result.index, pd.Index)
    assert result.shape == (16, 2)


def test_with_mixed_favorability(forest_binary_classification_data):
    """Test CV with mixed favorability indicators."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"], favorability=True)

    assert "favorability" in result.columns
    assert isinstance(result.index, pd.Index)
    assert result.loc["accuracy", "favorability"] == "(↗︎)"
    assert result.loc["brier_score", "favorability"] == "(↘︎)"


def test_frame_has_split_columns(forest_binary_classification_data):
    """The wide frame exposes one column per split via a MultiIndex."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    frame = report.metrics.summarize().frame(flat_index=False, aggregate=None)

    assert isinstance(frame.index, pd.Index)
    assert isinstance(frame.columns, pd.MultiIndex)
    assert frame.columns.names == ["estimator", "split"]
    assert set(frame.columns.get_level_values("split")) == {"Split #0", "Split #1"}


def test_wide_frame_aggregate_mean_std(forest_binary_classification_data):
    """Wide layout with aggregate exposes mean/std columns, not per-split columns."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    wide = display.frame(aggregate=["mean", "std"])
    wide_unflat = display.frame(flat_index=False, aggregate=["mean", "std"])

    assert "randomforestclassifier_mean" in wide.columns
    assert "randomforestclassifier_std" in wide.columns
    assert isinstance(wide_unflat.columns, pd.MultiIndex)
    assert wide_unflat.columns.names == ["estimator", "aggregate"]
    assert set(wide_unflat.columns.get_level_values("aggregate")) == {"mean", "std"}


def test_wide_frame_verbose_name_level_names(forest_binary_classification_data):
    """`verbose_name=True` capitalizes the row and column MultiIndex level names."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    result = report.metrics.summarize().frame(
        aggregate=["mean", "std"], flat_index=False, verbose_name=True
    )
    assert result.index.names == ["Metric", "Label"]
    assert result.columns.names == ["Estimator", "Aggregate"]


def test_plot_single_metric(forest_binary_classification_data):
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    fig = display.plot(metric="accuracy")
    assert isinstance(fig.axes[0], mpl.axes.Axes)
    assert fig._suptitle.get_text() == "Metrics of RandomForestClassifier"


@pytest.mark.parametrize(
    "fixture_name, metric, subplot_by, err_msg",
    [
        (
            "cross_validation_reports_binary_classification",
            "score",
            "label",
            r"Invalid `subplot_by` parameter\. Valid options are: auto, split, None\.",
        ),
        (
            "cross_validation_reports_regression",
            "score",
            "output",
            r"Invalid `subplot_by` parameter\. Valid options are: auto, split, None\.",
        ),
        (
            "cross_validation_reports_multiclass_classification",
            "precision",
            "incorrect",
            (
                r"Invalid `subplot_by` parameter\. Valid options are: "
                r"auto, label, split, None\."
            ),
        ),
        (
            "cross_validation_reports_multioutput_regression",
            "r2",
            "incorrect",
            (
                r"Invalid `subplot_by` parameter\. Valid options are: "
                r"auto, output, split, None\."
            ),
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, metric, subplot_by, err_msg, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.metrics.summarize(metric=metric)
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric=metric, subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, metric, subplot_by_tuples",
    [
        (
            "cross_validation_reports_binary_classification",
            "score",
            [(None, 1), ("split", 2)],
        ),
        (
            "cross_validation_reports_multiclass_classification",
            "precision",
            [("label", 3), (None, 1)],
        ),
        (
            "cross_validation_reports_regression",
            "score",
            [(None, 1)],
        ),
        (
            "cross_validation_reports_multioutput_regression",
            "r2",
            [("output", 2), (None, 1)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, metric, subplot_by_tuples, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.metrics.summarize(metric=metric)
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(metric=metric, subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len
