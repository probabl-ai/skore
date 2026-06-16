import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from skore import ComparisonReport, MetricsSummaryDisplay


def test_data_source_both(
    binary_classification_data, estimator_reports_binary_classification
):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame()

    assert result.index.to_list() == [
        ("Score", ""),
        ("Accuracy", ""),
        ("Precision", "0"),
        ("Precision", "1"),
        ("Recall", "0"),
        ("Recall", "1"),
        ("ROC AUC", ""),
        ("Log loss", ""),
        ("Brier score", ""),
        ("Fit time (s)", ""),
        ("Predict time (s)", ""),
    ]
    assert result.columns.to_list() == [
        "DummyClassifier_1 (train)",
        "DummyClassifier_1 (test)",
        "DummyClassifier_2 (train)",
        "DummyClassifier_2 (test)",
    ]


def test_flat_index(estimator_reports_binary_classification):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame(flat_index=True)
    assert isinstance(result_df.index, pd.Index)
    assert result_df.index.tolist() == [
        "score",
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
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_favorability(comparison_estimator_reports_binary_classification):
    """Check that the behaviour of `favorability` is correct."""
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.summarize()
    result = display.frame(favorability=True)
    assert set(result["Favorability"]) == {"(↗︎)", "(↘︎)"}


def test_aggregate(comparison_estimator_reports_binary_classification):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    np.testing.assert_allclose(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(),
    )


@pytest.mark.parametrize(
    "fixture_name, metric, valid_values",
    [
        (
            "comparison_estimator_reports_binary_classification",
            "score",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_estimator_reports_multiclass_classification",
            "precision",
            ["estimator", "label", "auto"],
        ),
        (
            "comparison_estimator_reports_regression",
            "score",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_estimator_reports_multioutput_regression",
            "r2",
            ["estimator", "output", "auto"],
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, metric, valid_values, request):
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.summarize()
    err_msg = (
        "Column incorrect not found in the frame."
        f" It should be one of {', '.join(valid_values)}."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric=metric, subplot_by="incorrect")


@pytest.mark.parametrize(
    "fixture_name, metric, subplot_by_tuples",
    [
        (
            "comparison_estimator_reports_binary_classification",
            "score",
            [(None, 1), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_multiclass_classification",
            "precision",
            [("label", 3), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_regression",
            "score",
            [(None, 1), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_multioutput_regression",
            "r2",
            [("output", 2), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(pyplot, fixture_name, metric, subplot_by_tuples, request):
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.summarize()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(metric=metric, subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len


@pytest.mark.parametrize(
    "fixture_name, metric",
    [
        ("comparison_estimator_reports_multiclass_classification", "precision"),
        ("comparison_estimator_reports_multioutput_regression", "r2"),
    ],
)
def test_subplot_by_none_multiclass_or_multioutput(
    pyplot,
    request,
    fixture_name,
    metric,
):
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.summarize()
    err_msg = (
        "There are multiple labels or outputs and `subplot_by` is `None`. "
        "There is too much information to display on a single plot. "
        "Please provide a column to group by using `subplot_by`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric=metric, subplot_by=None)
