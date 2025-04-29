"""Tests of `ComparisonReport.metrics.report_metrics`."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from skore import ComparisonReport, CrossValidationReport
from skore.utils._testing import check_cache_changed, check_cache_unchanged


@pytest.fixture
def classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    return X, y


def make_classifier():
    return DummyClassifier(strategy="uniform", random_state=0)


@pytest.fixture
def report(classification_data):
    """ComparisonReport of CrossValidationReports for classification estimators.

    Note that the two CrossValidationReports do not have the same number of CV splits.
    """
    X, y = classification_data

    report = ComparisonReport(
        [
            CrossValidationReport(make_classifier(), X, y),
            CrossValidationReport(make_classifier(), X, y, cv_splitter=3),
        ]
    )

    return report


@pytest.fixture
def report_regression():
    """ComparisonReport of CrossValidationReports for regression estimators."""
    X, y = make_regression(random_state=42)

    report = ComparisonReport(
        [
            CrossValidationReport(DummyRegressor(), X, y),
            CrossValidationReport(DummyRegressor(), X, y, cv_splitter=3),
        ]
    )

    return report


def test_aggregate_none(report):
    """`report_metrics` works as intended with `aggregate=None`."""
    result = report.metrics.report_metrics(aggregate=None)

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert result.index.names == ["Metric", "Label / Average", "Estimator", "Split"]
    assert len(result) == 64


def test_aggregate_none_flat_index(report):
    """
    `report_metrics` works as intended with `aggregate=None` and `flat_index=True`.
    """
    result = report.metrics.report_metrics(
        aggregate=None,
        flat_index=True,
    )

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert len(result) == 64


def test_default(report):
    """`report_metrics` works as intended with its default attributes."""
    result = report.metrics.report_metrics()

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
    assert len(result) == 8


def test_default_regression(report_regression):
    """
    `report_metrics` works as intended with its default attributes for regression
    models.
    """
    result = report_regression.metrics.report_metrics()

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
        pd.Index(["R²", "RMSE", "Fit time (s)", "Predict time (s)"], name="Metric"),
    )


def test_aggregate_sequence_of_one_element(report):
    """Passing a list of one string is the same as passing the string itself."""
    assert_frame_equal(
        report.metrics.report_metrics(aggregate="mean"),
        report.metrics.report_metrics(aggregate=["mean"]),
    )


def test_aggregate_is_used_in_cache(report):
    """`aggregate` should be used when computing the cache key.

    In other words, if you call `report_metrics` twice with different values of
    `aggregate`, you should get a different result.
    """
    call1 = report.metrics.report_metrics(aggregate="mean")
    call2 = report.metrics.report_metrics(aggregate=("mean", "std"))
    assert list(call1.columns) != list(call2.columns)


def test_scoring(report):
    """`report_metrics` works as intended with the `scoring` parameter."""
    result = report.metrics.report_metrics(
        scoring=["accuracy"],
        aggregate=None,
    )

    assert_index_equal(result.columns, pd.Index(["Value"]))
    assert_index_equal(
        result.index,
        pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "DummyClassifier_1", "Split #0"),
                ("Accuracy", "DummyClassifier_1", "Split #1"),
                ("Accuracy", "DummyClassifier_1", "Split #2"),
                ("Accuracy", "DummyClassifier_1", "Split #3"),
                ("Accuracy", "DummyClassifier_1", "Split #4"),
                ("Accuracy", "DummyClassifier_2", "Split #0"),
                ("Accuracy", "DummyClassifier_2", "Split #1"),
                ("Accuracy", "DummyClassifier_2", "Split #2"),
            ],
            names=("Metric", "Estimator", "Split"),
        ),
    )


def test_favorability(report):
    """`report_metrics` works as intended with `indicator_favorability=True`."""
    result = report.metrics.report_metrics(indicator_favorability=True)

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
    assert len(result) == 8


def test_cache(report):
    """`report_metrics` results are cached."""

    with check_cache_changed(report._cache):
        result = report.metrics.report_metrics()

    with check_cache_unchanged(report._cache):
        cached_result = report.metrics.report_metrics()

    assert_frame_equal(result, cached_result)


def test_init_with_report_names(classification_data):
    """
    If the estimators are passed as a dict, then the estimator names are the dict keys.
    """

    X, y = classification_data
    cv_report1 = CrossValidationReport(make_classifier(), X, y)
    cv_report2 = CrossValidationReport(make_classifier(), X, y)

    comp = ComparisonReport({"r1": cv_report1, "r2": cv_report2})

    assert_index_equal(
        (
            comp.metrics.report_metrics(aggregate=None)
            .index.get_level_values("Estimator")
            .unique()
        ),
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_X_y(report, classification_data):
    """`report_metrics` works as intended with `data_source="X_y"`."""
    X, y = classification_data
    result = report.metrics.report_metrics(data_source="X_y", X=X, y=y)

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
    assert len(result) == 8
