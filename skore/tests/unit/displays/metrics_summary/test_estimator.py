"""Tests for MetricsSummaryDisplay._to_pivoted_frame() method."""

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

    result_no_fav = display._to_pivoted_frame(favorability=False)
    assert result_no_fav.columns.to_list() == ["RandomForestClassifier"]

    result_with_fav = display._to_pivoted_frame(favorability=True)
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

    result_no_fav = display._to_pivoted_frame(favorability=False)
    assert result_no_fav.columns.to_list() == ["LinearRegression"]

    result_with_fav = display._to_pivoted_frame(favorability=True)
    assert result_with_fav.columns.to_list() == ["LinearRegression", "Favorability"]
    assert set(result_with_fav["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_flat_index_multiclass(forest_multiclass_classification_with_test):
    """Test flat_index parameter with multiclass classification data."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result_multi = display._to_pivoted_frame(favorability=False, flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Label"]

    result_flat = display._to_pivoted_frame(favorability=False, flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    assert result_flat.index.to_list() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "precision_2",
        "recall_0",
        "recall_1",
        "recall_2",
        "roc_auc_0",
        "roc_auc_1",
        "roc_auc_2",
        "log_loss",
        "fit_time_s",
        "predict_time_s",
    ]


def test_flat_index_multioutput(linear_regression_multioutput_with_test):
    """Test flat_index with multioutput regression data."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result_multi = display._to_pivoted_frame(favorability=False, flat_index=False)
    assert isinstance(result_multi.index, pd.MultiIndex)
    assert result_multi.index.names == ["Metric", "Output"]
    assert result_multi.shape == (10, 1)
    assert result_multi.loc[("R²", "0"), "LinearRegression"] == 1
    assert result_multi.loc[("R²", "1"), "LinearRegression"] == 1

    result_flat = display._to_pivoted_frame(favorability=False, flat_index=True)
    assert isinstance(result_flat.index, pd.Index)
    # Note: "R²" is lowercased
    assert result_flat.index.to_list() == [
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


def test_custom_macro_metric_uses_average(forest_binary_classification_with_test):
    """Average-only classification metrics should render in `Average`."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "Precision (Macro)"
    report.metrics.add(make_scorer(precision_score, average="macro"), name=name)
    result = report.metrics.summarize(metric=[name])._to_pivoted_frame()
    assert result.reset_index()["Average"].tolist() == ["macro"]


def test_multioutput_average_uses_output_average(
    linear_regression_multioutput_with_test,
):
    """Average-only multioutput regression metrics should render in `Average`."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "MAE (Average)"
    report.metrics.add(
        make_scorer(mean_absolute_error, multioutput="uniform_average"), name=name
    )
    result = report.metrics.summarize(metric=[name])._to_pivoted_frame()
    assert result.reset_index()["Average"].tolist().count("uniform_average") == 1


def test_flat_index_with_favorability(forest_binary_classification_with_test):
    """Test that flat_index and favorability work together."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display._to_pivoted_frame(favorability=True, flat_index=True)
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
        "fit_time_s",
        "predict_time_s",
    ]


def test_data_source_both_favorability(forest_binary_classification_data):
    """Test favorability with data_source='both' (train and test)."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    result_no_fav = display._to_pivoted_frame(favorability=False)
    assert result_no_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]

    result_with_fav = display._to_pivoted_frame(favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
        "Favorability",
    ]


def test_data_source_both_flat_index(forest_binary_classification_data):
    """Test flat_index with data_source='both' (train and test)."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    result = display._to_pivoted_frame(flat_index=True)
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
        "fit_time_s",
        "predict_time_s",
    ]


def test_frame_flat_index(forest_binary_classification_with_test):
    """The tidy frame has a flat index and the expected metadata columns."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame()

    assert isinstance(frame.index, pd.RangeIndex)
    assert not isinstance(frame.columns, pd.MultiIndex)
    # estimator is constant, so it is not exposed; no split nor data_source either
    assert frame.columns.to_list() == ["metric", "label", "value"]
    assert "Accuracy" in frame["metric"].to_numpy()


def test_frame_favorability_column(forest_binary_classification_with_test):
    """`favorability=True` appends a ``favorability`` column with arrow indicators."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    frame = report.metrics.summarize().frame()
    frame_fav = report.metrics.summarize().frame(favorability=True)

    assert "favorability" not in frame.columns
    assert frame_fav.columns.to_list() == [*frame.columns, "favorability"]
    assert set(frame_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_multiclass_has_label_column(forest_multiclass_classification_with_test):
    """Per-class metrics expose a ``label`` column in long format."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame()

    assert "label" in frame.columns
    assert "output" not in frame.columns
    precision = frame[frame["metric"] == "Precision"]
    assert precision["label"].to_list() == [0, 1, 2]


def test_frame_multioutput_has_output_column(linear_regression_multioutput_with_test):
    """Multioutput regression metrics expose an ``output`` column."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame()

    assert "output" in frame.columns
    assert "label" not in frame.columns
    r2 = frame[frame["metric"] == "R²"]
    assert r2["output"].to_list() == [0, 1]


def test_frame_data_source_both(forest_binary_classification_data):
    """With both data sources, the frame exposes a ``data_source`` column."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    frame = report.metrics.summarize(data_source="both").frame()

    assert "data_source" in frame.columns
    assert set(frame["data_source"]) == {"train", "test"}
