"""Tests for MetricsSummaryDisplay with an EstimatorReport."""

import matplotlib as mpl
import pandas as pd
import pytest
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

    result_no_fav = display.frame(favorability=False)
    assert isinstance(result_no_fav, pd.Series)
    assert result_no_fav.name == "RandomForestClassifier"
    assert result_no_fav.loc["accuracy"] is not None

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier",
        "favorability",
    ]
    assert set(result_with_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_favorability_regression(linear_regression_with_test):
    """Test that favorability column is correctly displayed for regression metrics."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result_no_fav = display.frame(favorability=False)
    assert isinstance(result_no_fav, pd.Series)
    assert result_no_fav.name == "LinearRegression"

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == ["LinearRegression", "favorability"]
    assert set(result_with_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_format_wide_multiclass(forest_multiclass_classification_with_test):
    """Compact format returns a flat index for multiclass classification."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display.frame(favorability=False)
    assert isinstance(result, pd.Series)
    assert result.name == "RandomForestClassifier"
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

    result = display.frame(favorability=False)
    assert isinstance(result, pd.Series)
    assert result.name == "LinearRegression"
    assert isinstance(result.index, pd.Index)
    assert len(result) == 10
    assert result.loc["r2_0"] == 1
    assert result.loc["r2_1"] == 1
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
    """Average-only classification metrics expose ``average`` in the row index."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "Precision (Macro)"
    report.metrics.add(make_scorer(precision_score, average="macro"), name=name)
    result = report.metrics.summarize(metric=[name]).frame(flat_index=False)
    assert result.index.names == ["metric", "average"]
    assert result.index.get_level_values("average").tolist() == ["macro"]


def test_multioutput_average_uses_output_average(
    linear_regression_multioutput_with_test,
):
    """Average-only multioutput regression metrics expose `average` in the row index."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    name = "MAE (Average)"
    report.metrics.add(
        make_scorer(mean_absolute_error, multioutput="uniform_average"), name=name
    )
    result = report.metrics.summarize(metric=[name]).frame(flat_index=False)
    assert result.index.names == ["metric", "average"]
    assert result.index.get_level_values("average").tolist() == ["uniform_average"]


def test_format_wide_with_favorability(forest_binary_classification_with_test):
    """Compact format and favorability work together."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    result = display.frame(favorability=True)
    assert result.columns.to_list() == ["RandomForestClassifier", "favorability"]

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

    result_no_fav = display.frame(favorability=False)
    assert result_no_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]

    result_with_fav = display.frame(favorability=True)
    assert result_with_fav.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
        "favorability",
    ]


def test_data_source_both_format_wide(forest_binary_classification_data):
    """Compact format with data_source='both' (train and test)."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    result = display.frame()
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
    """The wide frame preserves metric and label levels in the row index."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(flat_index=False)

    assert isinstance(frame, pd.Series)
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.index.names == ["metric", "label"]
    assert "accuracy" in frame.index.get_level_values("metric").to_numpy()


def test_frame_favorability_column(forest_binary_classification_with_test):
    """`favorability=True` appends a ``favorability`` column with arrow indicators."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    frame = report.metrics.summarize().frame(flat_index=False)
    frame_fav = report.metrics.summarize().frame(favorability=True, flat_index=False)

    assert isinstance(frame, pd.Series)
    assert frame_fav.columns.to_list() == ["RandomForestClassifier", "favorability"]
    assert set(frame_fav["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_multiclass_has_label_column(forest_multiclass_classification_with_test):
    """Per-class metrics expose a ``label`` level in the row index."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(flat_index=False)

    assert "label" in frame.index.names
    assert "average" in frame.index.names
    assert "output" not in frame.index.names
    precision_index = frame.loc["precision"].index
    precision_labels = [
        label
        for label, average in zip(
            precision_index.get_level_values("label"),
            precision_index.get_level_values("average"),
            strict=True,
        )
        if average == ""
    ]
    assert [int(label) for label in precision_labels] == [0, 1, 2]


def test_frame_multiclass_includes_macro_metrics(
    forest_multiclass_classification_with_test,
):
    """Built-in macro metrics are exposed via the ``average`` dimension."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    data = report.metrics.summarize().summary

    for metric_name in ("precision", "recall", "roc_auc"):
        macro_rows = data[(data["name"] == metric_name) & (data["average"] == "macro")]
        assert len(macro_rows) == 1


def test_frame_multioutput_has_output_column(linear_regression_multioutput_with_test):
    """Multioutput regression metrics expose an ``output`` level in the row index."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    frame = report.metrics.summarize().frame(flat_index=False)

    assert "output" in frame.index.names
    assert "label" not in frame.index.names
    assert frame.loc["r2"].index.get_level_values("output").to_list() == [0, 1]


def test_frame_verbose_name_true(forest_binary_classification_with_test):
    """`verbose_name=True` exposes human-readable metric names in the frame."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    frame = report.metrics.summarize().frame(verbose_name=True, flat_index=False)

    assert "Metric" in frame.index.names
    assert "Accuracy" in frame.index.get_level_values("Metric").to_numpy()
    assert "accuracy" not in frame.index.get_level_values("Metric").to_numpy()


def test_frame_verbose_name_true_wide(forest_binary_classification_with_test):
    """`verbose_name=True` uses verbose-derived names in wide format."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize().frame(
        verbose_name=True, flat_index=False, favorability=False
    )

    assert ("Fit time (s)", "") in result.index
    assert ("fit_time", "") not in result.index
    assert ("Accuracy", "") in result.index
    assert ("accuracy", "") not in result.index


def test_frame_flat_index_false(forest_multiclass_classification_with_test):
    """`flat_index=False` preserves row MultiIndex in wide format."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.summarize().frame(
        flat_index=False, favorability=False, verbose_name=True
    )

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Label", "Average"]
    assert "Output" not in result.index.names
    assert ("Precision", "", "macro") in result.index


def test_frame_data_source_both(forest_binary_classification_data):
    """With both data sources, the frame exposes train and test value columns."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    frame = report.metrics.summarize(data_source="both").frame(flat_index=False)

    assert frame.columns.to_list() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]


def test_plot_single_metric(pyplot, forest_binary_classification_with_test):
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    fig = display.plot(metric="accuracy")
    assert isinstance(fig.axes[0], mpl.axes.Axes)
    assert fig._suptitle.get_text() == "Metrics of RandomForestClassifier"


def test_plot_unknown_metric_raises(forest_binary_classification_with_test):
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    with pytest.raises(ValueError, match="Unknown metric"):
        display.plot(metric="not_a_metric")


def test_plot_data_source_both(pyplot, forest_binary_classification_data):
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.summarize(data_source="both")

    fig = display.plot(metric="accuracy")
    assert len(fig.axes) == 1


@pytest.mark.parametrize(
    "fixture_name, metric, subplot_by, err_msg",
    [
        (
            "estimator_reports_binary_classification",
            "score",
            "label",
            "No columns to group by.",
        ),
        (
            "estimator_reports_regression",
            "score",
            "output",
            "No columns to group by.",
        ),
        (
            "estimator_reports_multiclass_classification",
            "precision",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of label, auto, None.",
        ),
        (
            "estimator_reports_multioutput_regression",
            "r2",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of output, auto, None.",
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
            "estimator_reports_binary_classification",
            "score",
            [(None, 1)],
        ),
        (
            "estimator_reports_multiclass_classification",
            "precision",
            [("label", 3), (None, 1)],
        ),
        (
            "estimator_reports_regression",
            "score",
            [(None, 1)],
        ),
        (
            "estimator_reports_multioutput_regression",
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
