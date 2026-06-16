"""Tests of `ComparisonReport.metrics.summarize`."""

import matplotlib as mpl
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.dummy import DummyClassifier

from skore import ComparisonReport, CrossValidationReport


def test_aggregate_none(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(aggregate=None)

    assert result.columns.to_list() == [
        ("DummyClassifier_1", "Split #0"),
        ("DummyClassifier_1", "Split #1"),
        ("DummyClassifier_2", "Split #0"),
        ("DummyClassifier_2", "Split #1"),
    ]
    assert result.columns.names == ["Estimator", "Split"]
    assert len(result) == 11
    assert isinstance(result.columns, pd.MultiIndex)


def test_aggregate_none_flat_index(
    comparison_cross_validation_reports_binary_classification,
):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None` and
    `flat_index=True`.
    """
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(aggregate=None, flat_index=True)

    assert result.columns.to_list() == [
        "dummyclassifier_1_split_0",
        "dummyclassifier_1_split_1",
        "dummyclassifier_2_split_0",
        "dummyclassifier_2_split_1",
    ]
    assert len(result) == 11


def test_default(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with its default attributes."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame()

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyClassifier_1"),
                ("mean", "DummyClassifier_2"),
                ("std", "DummyClassifier_1"),
                ("std", "DummyClassifier_2"),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert len(result) == 11


def test_default_regression(comparison_cross_validation_reports_regression):
    """
    `MetricsSummaryDisplay` works as intended with its default attributes for regression
    models.
    """
    report = comparison_cross_validation_reports_regression
    result = report.metrics.summarize().frame()

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyRegressor_1"),
                ("mean", "DummyRegressor_2"),
                ("std", "DummyRegressor_1"),
                ("std", "DummyRegressor_2"),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert_index_equal(
        result.index,
        pd.Index(
            ["Score", "R²", "RMSE", "MAE", "MAPE", "Fit time (s)", "Predict time (s)"],
            name="Metric",
        ),
    )


def test_aggregate_none_regression(comparison_cross_validation_reports_regression):
    """`MetricsSummaryDisplay` works as it should with `aggregate=None` for
    regression."""
    report = comparison_cross_validation_reports_regression
    result = report.metrics.summarize().frame(aggregate=None)

    assert result.columns.names == ["Estimator", "Split"]
    assert isinstance(result.columns, pd.MultiIndex)
    assert len(result) == 7


def test_metric(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with the `metric` parameter."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(metric=["accuracy"]).frame(aggregate=None)

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("DummyClassifier_1", "Split #0"),
                ("DummyClassifier_1", "Split #1"),
                ("DummyClassifier_2", "Split #0"),
                ("DummyClassifier_2", "Split #1"),
            ],
            names=["Estimator", "Split"],
        ),
    )
    assert len(result) == 1


def test_favorability(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `favorability=True`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize().frame(favorability=True)

    assert_index_equal(
        result.columns,
        pd.MultiIndex.from_tuples(
            [
                ("mean", "DummyClassifier_1"),
                ("mean", "DummyClassifier_2"),
                ("std", "DummyClassifier_1"),
                ("std", "DummyClassifier_2"),
                ("Favorability", ""),
            ],
            names=[None, "Estimator"],
        ),
    )
    assert len(result) == 11


def test_init_with_report_names(binary_classification_data):
    """
    If the estimators are passed as a dict, then the estimator names are the dict keys.
    """
    X, y = binary_classification_data

    report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y
    )
    report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=2), X=X, y=y
    )
    report = ComparisonReport({"model_1": report_1, "model_2": report_2})

    estimator_names = set(
        report.metrics.summarize()
        .frame(aggregate=None)
        .columns.get_level_values("Estimator")
    )
    assert estimator_names == {"model_1", "model_2"}


def test_cache_poisoning(binary_classification_data):
    """
    Computing metrics for a ComparisonReport should not influence the
    metrics computation for the internal CVReports.

    Non-regression test for https://github.com/probabl-ai/skore/issues/1706
    """
    X, y = binary_classification_data

    report_1 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=1), X=X, y=y
    )
    report_2 = CrossValidationReport(
        DummyClassifier(strategy="uniform", random_state=2), X=X, y=y
    )
    report = ComparisonReport({"model_1": report_1, "model_2": report_2})
    report.metrics.summarize().frame(favorability=True)
    result = report_1.metrics.summarize().frame(aggregate=None, favorability=True)

    assert "Favorability" in result.columns


def test_aggregate_sequence_of_one_element(
    comparison_cross_validation_reports_binary_classification,
):
    """Passing a list of one string is the same as passing the string itself."""
    report = comparison_cross_validation_reports_binary_classification
    assert_frame_equal(
        report.metrics.summarize().frame(aggregate="mean"),
        report.metrics.summarize().frame(aggregate=["mean"]),
    )


@pytest.mark.parametrize(
    "fixture_name, metric, valid_values",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            "score",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            "precision",
            ["estimator", "label", "auto"],
        ),
        (
            "comparison_cross_validation_reports_regression",
            "score",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_cross_validation_reports_multioutput_regression",
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
            "comparison_cross_validation_reports_binary_classification",
            "score",
            [(None, 1), ("estimator", 2)],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            "precision",
            [("label", 3), ("estimator", 2)],
        ),
        (
            "comparison_cross_validation_reports_regression",
            "score",
            [(None, 1), ("estimator", 2)],
        ),
        (
            "comparison_cross_validation_reports_multioutput_regression",
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
        ("comparison_cross_validation_reports_multiclass_classification", "precision"),
        ("comparison_cross_validation_reports_multioutput_regression", "r2"),
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
