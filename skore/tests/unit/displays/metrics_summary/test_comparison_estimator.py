import matplotlib as mpl
import pandas as pd
import pytest

from skore import ComparisonReport, MetricsSummaryDisplay


def test_default_is_wide(estimator_reports_binary_classification):
    """Default ``frame()`` returns wide layout for comparison-estimator reports."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])

    result = report.metrics.summarize().frame()

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.Index)
    assert result.columns.tolist() == ["DummyClassifier_1", "DummyClassifier_2"]


def test_data_source_both(estimator_reports_binary_classification):
    """Check that `MetricsSummaryDisplay` works with `data_source="both"`."""
    estimator_report_1, estimator_report_2 = estimator_reports_binary_classification
    report = ComparisonReport([estimator_report_1, estimator_report_2])
    result = report.metrics.summarize(data_source="both").frame()

    assert result.index.to_list() == [
        "score",
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
    assert result.columns.to_list() == [
        "DummyClassifier_1 (train)",
        "DummyClassifier_1 (test)",
        "DummyClassifier_2 (train)",
        "DummyClassifier_2 (test)",
    ]


def test_format_wide(estimator_reports_binary_classification):
    """Compact format always returns a flat index and columns."""
    report_1, report_2 = estimator_reports_binary_classification
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)
    result_df = result.frame()
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
        "fit_time",
        "predict_time",
    ]
    assert result_df.columns.tolist() == ["report_1", "report_2"]


def test_favorability(comparison_estimator_reports_binary_classification):
    """Check that the behaviour of `favorability` is correct."""
    report = comparison_estimator_reports_binary_classification
    display = report.metrics.summarize()
    result = display.frame(favorability=True)
    assert set(result["favorability"]) == {"(↗︎)", "(↘︎)"}


def test_frame_has_estimator_columns(
    comparison_estimator_reports_binary_classification,
):
    """The wide frame exposes one column per compared estimator."""
    report = comparison_estimator_reports_binary_classification
    frame = report.metrics.summarize().frame(flat_index=False)

    assert isinstance(frame.index, pd.Index)
    assert frame.columns.tolist() == ["DummyClassifier_1", "DummyClassifier_2"]
    assert "split" not in frame.columns


def test_aggregate(comparison_estimator_reports_binary_classification):
    """Passing `aggregate` should have no effect, as this argument is only relevant
    when comparing `CrossValidationReport`s."""
    report = comparison_estimator_reports_binary_classification
    from pandas.testing import assert_frame_equal

    assert_frame_equal(
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
def test_invalid_subplot_by(fixture_name, metric, valid_values, request):
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
def test_valid_subplot_by(fixture_name, metric, subplot_by_tuples, request):
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
