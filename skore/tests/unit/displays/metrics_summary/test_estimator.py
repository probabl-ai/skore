"""Tests for MetricsSummaryDisplay.frame() method with bare DataFrames.

These tests focus on testing the display/formatting logic of MetricsSummaryDisplay
without depending on EstimatorReport or summarize().
"""

import pandas as pd

from skore._sklearn._plot.metrics.metrics_summary_display import MetricsSummaryDisplay


def test_frame_favorability_binary_classification():
    """
    Test that favorability column is correctly displayed for binary classification.
    """
    data = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "precision",
                "recall",
                "recall",
                "roc_auc",
                "brier_score",
            ],
            "verbose_name": [
                "Accuracy",
                "Precision",
                "Precision",
                "Recall",
                "Recall",
                "ROC AUC",
                "Brier score",
            ],
            "label": ["", "0", "1", "0", "1", "", ""],
            "estimator_name": ["RandomForestClassifier"] * 7,
            "score": [0.95, 0.92, 0.97, 0.89, 0.98, 0.96, 0.08],
            "favorability": ["(↗︎)", "(↗︎)", "(↗︎)", "(↗︎)", "(↗︎)", "(↗︎)", "(↘︎)"],
            "data_source": ["test"] * 7,
            "average": [None] * 7,
            "output": [None] * 7,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result_no_fav = display.frame(favorability=False)
    assert result_no_fav.columns.to_list() == ["RandomForestClassifier"]

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier",
        "Favorability",
    ]
    assert set(result_with_fav["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_favorability_regression():
    """Test that favorability column is correctly displayed for regression metrics."""
    data = pd.DataFrame(
        {
            "metric": ["r2", "rmse"],
            "verbose_name": ["R²", "RMSE"],
            "estimator_name": ["LinearRegression", "LinearRegression"],
            "score": [0.85, 0.15],
            "favorability": ["(↗︎)", "(↘︎)"],
            "data_source": ["test", "test"],
            "average": [None] * 2,
            "output": [None] * 2,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result_no_fav = display.frame(favorability=False)
    assert result_no_fav.columns.to_list() == ["LinearRegression"]

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == ["LinearRegression", "Favorability"]
    assert set(result_with_fav["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_flat_index_multiclass():
    """Test flat_index parameter with multiclass classification data."""
    data = pd.DataFrame(
        {
            "metric": ["precision"] * 3 + ["recall"] * 3,
            "verbose_name": ["Precision"] * 3 + ["Recall"] * 3,
            "label": ["0", "1", "2"] * 2,
            "estimator_name": ["RandomForestClassifier"] * 6,
            "score": [0.92, 0.87, 0.91, 0.88, 0.85, 0.90],
            "favorability": ["(↗︎)"] * 6,
            "data_source": ["test"] * 6,
            "average": [None] * 6,
            "output": [None] * 6,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result_multi = display.frame(favorability=False, flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Label / Average"]

    result_flat = display.frame(favorability=False, flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    assert result_flat.index.to_list() == [
        "precision_0",
        "precision_1",
        "precision_2",
        "recall_0",
        "recall_1",
        "recall_2",
    ]


def test_frame_flat_index_with_favorability():
    """Test that flat_index and favorability work together."""
    data = pd.DataFrame(
        {
            "metric": ["precision", "precision", "recall", "recall"],
            "verbose_name": ["Precision", "Precision", "Recall", "Recall"],
            "label": ["0", "1", "0", "1"],
            "estimator_name": ["LogisticRegression"] * 4,
            "score": [0.85, 0.90, 0.88, 0.92],
            "favorability": ["(↗︎)"] * 4,
            "data_source": ["test"] * 4,
            "average": [None] * 4,
            "output": [None] * 4,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result = display.frame(favorability=True, flat_index=True)
    assert result.columns.to_list() == ["LogisticRegression", "Favorability"]

    assert isinstance(result.index, pd.Index)
    assert result.index.to_list() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
    ]


def test_frame_data_source_both_with_favorability():
    """Test favorability with data_source='both' (train and test)."""
    data = pd.DataFrame(
        {
            "metric": ["accuracy", "accuracy", "roc_auc", "roc_auc"],
            "verbose_name": ["Accuracy", "Accuracy", "ROC AUC", "ROC AUC"],
            "label": [None] * 4,
            "estimator_name": ["RandomForestClassifier"] * 4,
            "score": [0.98, 0.95, 0.99, 0.96],
            "favorability": ["(↗︎)", "(↗︎)", "(↗︎)", "(↗︎)"],
            "data_source": ["train", "test", "train", "test"],
            "average": [None] * 4,
            "output": [None] * 4,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result_no_fav = display.frame(favorability=False)
    assert result_no_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
        "Favorability",
    ]


def test_frame_multioutput_with_flat_index():
    """Test flat_index with multioutput regression data."""
    data = pd.DataFrame(
        {
            "metric": ["r2", "r2", "r2", "rmse", "rmse", "rmse"],
            "verbose_name": ["R²", "R²", "R²", "RMSE", "RMSE", "RMSE"],
            "output": ["0", "1", "2", "0", "1", "2"],
            "estimator_name": ["LinearRegression"] * 6,
            "score": [0.85, 0.78, 0.92, 0.12, 0.18, 0.09],
            "favorability": ["(↗︎)", "(↗︎)", "(↗︎)", "(↘︎)", "(↘︎)", "(↘︎)"],
            "data_source": ["test"] * 6,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")

    result_multi = display.frame(favorability=False, flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Output"]

    result_flat = display.frame(favorability=False, flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    # Note: "R²" is lowercased
    assert result_flat.index.to_list() == [
        "r²_0",
        "r²_1",
        "r²_2",
        "rmse_0",
        "rmse_1",
        "rmse_2",
    ]


def test_frame_multioutput_multiindex():
    """Test that multioutput data creates proper MultiIndex."""
    data = pd.DataFrame(
        {
            "metric": ["r2", "r2", "rmse", "rmse"],
            "verbose_name": ["R²", "R²", "RMSE", "RMSE"],
            "output": ["0", "1", "0", "1"],
            "estimator_name": ["LinearRegression"] * 4,
            "score": [0.85, 0.78, 0.12, 0.18],
            "favorability": ["(↗︎)", "(↗︎)", "(↘︎)", "(↘︎)"],
            "data_source": ["test"] * 4,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")
    result = display.frame(favorability=False)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Output"]
    assert result.shape == (4, 1)
    assert result.loc[("R²", "0"), "LinearRegression"] == 0.85
    assert result.loc[("R²", "1"), "LinearRegression"] == 0.78


def test_frame_single_data_source_dropped():
    """Test that data_source column is dropped when there's only one source."""
    data = pd.DataFrame(
        {
            "metric": ["accuracy", "precision"],
            "verbose_name": ["Accuracy", "Precision"],
            "estimator_name": ["LogisticRegression"] * 2,
            "score": [0.95, 0.92],
            "favorability": ["(↗︎)", "(↗︎)"],
            "data_source": ["test", "test"],
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")
    result = display.frame(favorability=False)

    assert "data_source" not in result.columns


def test_frame_multiple_data_sources_pivoted():
    """Test that multiple data sources create separate columns."""
    data = pd.DataFrame(
        {
            "metric": ["accuracy", "accuracy"],
            "verbose_name": ["Accuracy", "Accuracy"],
            "estimator_name": ["LogisticRegression"] * 2,
            "score": [0.98, 0.95],
            "favorability": ["(↗︎)", "(↗︎)"],
            "data_source": ["train", "test"],
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")
    result = display.frame(favorability=False)

    assert result.columns.to_list() == [
        "LogisticRegression (train)",
        "LogisticRegression (test)",
    ]
    assert result.loc["Accuracy", "LogisticRegression (train)"] == 0.98
    assert result.loc["Accuracy", "LogisticRegression (test)"] == 0.95


def test_frame_empty_label_values():
    """Test that empty label values are handled correctly."""
    data = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "precision"],
            "verbose_name": ["Accuracy", "Precision", "Precision"],
            "label": ["", "0", "1"],
            "estimator_name": ["RandomForestClassifier"] * 3,
            "score": [0.95, 0.92, 0.97],
            "favorability": ["(↗︎)", "(↗︎)", "(↗︎)"],
            "data_source": ["test"] * 3,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")
    result = display.frame(favorability=False)

    # Empty string labels should be preserved in the MultiIndex
    assert isinstance(result.index, pd.MultiIndex)
    assert ("Accuracy", "") in result.index
    assert ("Precision", "0") in result.index
    assert ("Precision", "1") in result.index


def test_frame_preserves_score_values():
    """Test that score values are preserved correctly in the output."""
    expected_scores = [0.95, 0.92, 0.87, 0.91]
    data = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1"],
            "verbose_name": ["Accuracy", "Precision", "Recall", "F1"],
            "estimator_name": ["SVC"] * 4,
            "score": expected_scores,
            "favorability": ["(↗︎)"] * 4,
            "data_source": ["test"] * 4,
        }
    )

    display = MetricsSummaryDisplay(data=data, report_type="estimator")
    result = display.frame(favorability=False)

    assert result["SVC"].tolist() == expected_scores
