"""Tests for MetricsSummaryDisplay with a CrossValidationReport."""

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
    assert result_multi.index.names == ["Metric", "Label"]

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
    display = report.metrics.summarize()

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
        "mae_0",
        "mae_1",
        "mape_0",
        "mape_1",
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


def test_data_source_both_favorability(forest_binary_classification_data):
    """Test favorability columns when data_source='both'."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    name = report.estimator_name_
    display = report.metrics.summarize(data_source="both")

    result = display.frame(favorability=False)
    assert result.columns.tolist() == [
        (f"{name} (train)", "mean"),
        (f"{name} (train)", "std"),
        (f"{name} (test)", "mean"),
        (f"{name} (test)", "std"),
    ]

    result = display.frame(favorability=True)
    assert result.columns.tolist() == [
        (f"{name} (train)", "mean"),
        (f"{name} (train)", "std"),
        (f"{name} (test)", "mean"),
        (f"{name} (test)", "std"),
        ("Favorability", ""),
    ]


def test_data_source_both_flat_index(forest_binary_classification_data):
    """Test flat_index columns and index when data_source='both'."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    name = report.estimator_name_.lower()
    result = report.metrics.summarize(data_source="both").frame(flat_index=True)

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
    assert result.index.names == ["Metric", "Label"]
    assert result.shape == (13, 2)


def test_with_mixed_favorability(forest_binary_classification_data):
    """Test CV with mixed favorability indicators."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    result = display.frame(aggregate=["mean", "std"], favorability=True)

    assert ("Favorability", "") in result.columns
    assert result.index.names == ["Metric", "Label"]
    assert result.loc[("Accuracy", ""), ("Favorability", "")] == "(↗︎)"
    assert result.loc[("Brier score", ""), ("Favorability", "")] == "(↘︎)"


def test_plot_single_metric(pyplot, forest_binary_classification_data):
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
            "No columns to group by.",
        ),
        (
            "cross_validation_reports_regression",
            "score",
            "output",
            "No columns to group by.",
        ),
        (
            "cross_validation_reports_multiclass_classification",
            "precision",
            "incorrect",
            "Column incorrect not found in the frame."
            + " It should be one of label, auto, None.",
        ),
        (
            "cross_validation_reports_multioutput_regression",
            "r2",
            "incorrect",
            "Column incorrect not found in the frame."
            + " It should be one of output, auto, None.",
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, metric, subplot_by, err_msg, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.metrics.summarize()
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric=metric, subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, metric, subplot_by_tuples",
    [
        (
            "cross_validation_reports_binary_classification",
            "score",
            [(None, 1)],
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
def test_valid_subplot_by(pyplot, fixture_name, metric, subplot_by_tuples, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.metrics.summarize()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(metric=metric, subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len
