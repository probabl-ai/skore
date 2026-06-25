"""Tests of `ComparisonReport.metrics.summarize`."""

import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.dummy import DummyClassifier

from skore import ComparisonReport, CrossValidationReport


def test_aggregate_none(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with `aggregate=None`."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize()._to_pivoted_frame(aggregate=None)

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
    result = report.metrics.summarize()._to_pivoted_frame(
        aggregate=None, flat_index=True
    )

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
    result = report.metrics.summarize()._to_pivoted_frame()

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
    result = report.metrics.summarize()._to_pivoted_frame()

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
    result = report.metrics.summarize()._to_pivoted_frame(aggregate=None)

    assert result.columns.names == ["Estimator", "Split"]
    assert isinstance(result.columns, pd.MultiIndex)
    assert len(result) == 7


def test_metric(comparison_cross_validation_reports_binary_classification):
    """`MetricsSummaryDisplay` works as intended with the `metric` parameter."""
    report = comparison_cross_validation_reports_binary_classification
    result = report.metrics.summarize(metric=["accuracy"])._to_pivoted_frame(
        aggregate=None
    )

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
    result = report.metrics.summarize()._to_pivoted_frame(favorability=True)

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
        ._to_pivoted_frame(aggregate=None)
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
    report.metrics.summarize()._to_pivoted_frame(favorability=True)
    result = report_1.metrics.summarize()._to_pivoted_frame(
        aggregate=None, favorability=True
    )

    assert "Favorability" in result.columns


def test_aggregate_sequence_of_one_element(
    comparison_cross_validation_reports_binary_classification,
):
    """Passing a list of one string is the same as passing the string itself."""
    report = comparison_cross_validation_reports_binary_classification
    assert_frame_equal(
        report.metrics.summarize()._to_pivoted_frame(aggregate="mean"),
        report.metrics.summarize()._to_pivoted_frame(aggregate=["mean"]),
    )


def test_frame_has_estimator_and_split_columns(
    comparison_cross_validation_reports_binary_classification,
):
    """The tidy frame exposes both ``estimator`` and ``split`` columns."""
    report = comparison_cross_validation_reports_binary_classification
    frame = report.metrics.summarize().frame()

    assert isinstance(frame.index, pd.RangeIndex)
    assert {"estimator", "split"}.issubset(frame.columns)
    assert frame["estimator"].nunique() == 2
    assert set(frame["split"]) == {0, 1}
