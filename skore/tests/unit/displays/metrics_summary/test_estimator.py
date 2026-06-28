"""Tests for MetricsSummaryDisplay.frame(format="wide") method."""

import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, precision_score
from sklearn.model_selection import train_test_split

from skore import EstimatorReport


def test_favorability_binary(forest_binary_classification_with_test):
    """
    Test that favorability column is correctly displayed for binary classification.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result_no_fav = display.frame(format="wide", favorability=False)
    assert result_no_fav.columns.to_list() == ["RandomForestClassifier"]

    result_with_fav = display.frame(format="wide", favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier",
        "Favorability",
    ]
    assert set(result_with_fav["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_favorability_regression(linear_regression_with_test):
    """Test that favorability column is correctly displayed for regression metrics."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result_no_fav = display.frame(format="wide", favorability=False)
    assert result_no_fav.columns.to_list() == ["LinearRegression"]

    result_with_fav = display.frame(format="wide", favorability=True)
    assert result_with_fav.columns.to_list() == ["LinearRegression", "Favorability"]
    assert set(result_with_fav["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_format_wide_multiclass(forest_multiclass_classification_with_test):
    """Compact format returns a flat index for multiclass classification."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display.frame(format="wide", favorability=False)
    assert isinstance(result.index, pd.Index)
    assert result.index.to_list() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "precision_2",
        "precision_macro",
        "recall_0",
        "recall_1",
        "recall_2",
        "recall_macro",
        "roc_auc_0",
        "roc_auc_1",
        "roc_auc_2",
        "roc_auc_macro",
        "log_loss",
        "fit_time",
        "predict_time",
    ]


def test_format_wide_multioutput(linear_regression_multioutput_with_test):
    """Compact format returns a flat index for multioutput regression."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display.frame(format="wide", favorability=False)
    assert isinstance(result.index, pd.Index)
    assert result.shape == (10, 1)
    assert result.loc["r2_0", "LinearRegression"] == 1
    assert result.loc["r2_1", "LinearRegression"] == 1
    assert result.index.to_list() == [
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


def test_custom_macro_metric_uses_average(forest_binary_classification_with_test):
    """Average-only classification metrics expose ``average`` in long format."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "Precision (Macro)"
    report.metrics.add(make_scorer(precision_score, average="macro"), name=name)
    result = report.metrics.summarize(metric=[name]).frame(format="long")
    assert result["average"].tolist() == ["macro"]


def test_multioutput_average_uses_output_average(
    linear_regression_multioutput_with_test,
):
    """Average-only multioutput regression metrics expose ``average`` in long format."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "MAE (Average)"
    report.metrics.add(
        make_scorer(mean_absolute_error, multioutput="uniform_average"), name=name
    )
    result = report.metrics.summarize(metric=[name]).frame(format="long")
    assert result["average"].tolist().count("uniform_average") == 1


def test_format_wide_with_favorability(forest_binary_classification_with_test):
    """Compact format and favorability work together."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display.frame(format="wide", favorability=True)
    assert result.columns.to_list() == ["RandomForestClassifier", "Favorability"]

    assert isinstance(result.index, pd.Index)
    assert result.index.to_list() == [
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
    """Test favorability with data_source='both' (train and test)."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    result_no_fav = display.frame(format="wide", favorability=False)
    assert result_no_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]

    result_with_fav = display.frame(format="wide", favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
        "Favorability",
    ]


def test_data_source_both_format_wide(forest_binary_classification_data):
    """Compact format with data_source='both' (train and test)."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    result = display.frame(format="wide")
    assert result.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]
    assert result.index.to_list() == [
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


def test_frame_flat_index(forest_binary_classification_with_test):
    """The tidy frame has a flat index and the expected metadata columns."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(format="long")

    assert isinstance(frame.index, pd.RangeIndex)
    assert not isinstance(frame.columns, pd.MultiIndex)
    # estimator is constant, so it is not exposed; no split nor data_source either
    assert frame.columns.to_list() == ["metric", "label", "value"]
    assert "accuracy" in frame["metric"].to_numpy()


def test_frame_favorability_column(forest_binary_classification_with_test):
    """`favorability=True` appends a ``favorability`` column with arrow indicators."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    frame = report.metrics.summarize().frame(format="long")
    frame_fav = report.metrics.summarize().frame(favorability=True, format="long")

    assert "favorability" not in frame.columns
    assert frame_fav.columns.to_list() == [*frame.columns, "favorability"]
    assert set(frame_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_multiclass_has_label_column(forest_multiclass_classification_with_test):
    """Per-class metrics expose a ``label`` column in long format."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(format="long")

    assert "label" in frame.columns
    assert "output" not in frame.columns
    precision = frame[frame["metric"] == "precision"]
    assert precision["label"].dropna().to_list() == [0, 1, 2]


def test_frame_multiclass_includes_aggregate_average_rows(
    forest_multiclass_classification_with_test,
):
    """Built-in precision/recall/roc_auc expose a macro aggregate row."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    data = report.metrics.summarize().summary

    for metric_name in ("precision", "recall", "roc_auc"):
        aggregate = data[
            (data["name"] == metric_name)
            & data["label"].isna()
            & data["average"].notna()
        ]
        assert set(aggregate["average"].tolist()) == {"macro"}


def test_frame_multioutput_has_output_column(linear_regression_multioutput_with_test):
    """Multioutput regression metrics expose an ``output`` column."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(format="long")

    assert "output" in frame.columns
    assert "label" not in frame.columns
    r2 = frame[frame["metric"] == "r2"]
    assert r2["output"].to_list() == [0, 1]


def test_frame_verbose_name_true(forest_binary_classification_with_test):
    """`verbose_name=True` exposes human-readable metric names in the frame."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    frame = report.metrics.summarize().frame(verbose_name=True, format="long")

    assert "Accuracy" in frame["metric"].to_numpy()
    assert "accuracy" not in frame["metric"].to_numpy()


def test_frame_verbose_name_true_wide(forest_binary_classification_with_test):
    """`verbose_name=True` uses verbose-derived names in wide format."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize().frame(
        format="wide", verbose_name=True, favorability=False
    )

    assert "Fit_time_s" in result.index
    assert "fit_time" not in result.index
    assert "Accuracy" in result.index
    assert "accuracy" not in result.index


def test_frame_with_multiindex(forest_multiclass_classification_with_test):
    """`with_multiindex=True` preserves row MultiIndex in wide format."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize().frame(
        format="wide", with_multiindex=True, favorability=False
    )

    assert isinstance(result.index, pd.MultiIndex)
    assert "Metric" in result.index.names


def test_frame_data_source_both(forest_binary_classification_data):
    """With both data sources, the frame exposes a ``data_source`` column."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    frame = report.metrics.summarize(data_source="both").frame(format="long")

    assert "data_source" in frame.columns
    assert set(frame["data_source"]) == {"train", "test"}
