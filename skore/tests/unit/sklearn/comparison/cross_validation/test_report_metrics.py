"""Tests of `ComparisonReport.metrics.report_metrics`."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from skore import ComparisonReport, CrossValidationReport


@pytest.fixture
def classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    return X, y


def make_classifier():
    return DummyClassifier(strategy="uniform", random_state=0)


@pytest.fixture
def report(classification_data):
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
    X, y = make_regression(random_state=42)

    report = ComparisonReport(
        [
            CrossValidationReport(DummyRegressor(), X, y),
            CrossValidationReport(DummyRegressor(), X, y, cv_splitter=3),
        ]
    )

    return report


def test_different_split_numbers(report):
    result = report.metrics.report_metrics(aggregate=None)

    assert_index_equal(
        result.columns,
        pd.Index(["DummyClassifier_1", "DummyClassifier_2"], name="Estimator"),
    )
    assert result.index.names == ["Metric", "Label / Average", "Split"]
    assert len(result) == ((2 * 2) + (4 * 1)) * 5


def test_flat_index_different_split_numbers(report):
    result = report.metrics.report_metrics(
        aggregate=None,
        flat_index=True,
    )

    assert_index_equal(
        result.columns,
        pd.Index(["DummyClassifier_1", "DummyClassifier_2"], name="Estimator"),
    )
    assert len(result) == ((2 * 2) + (4 * 1)) * 5


def test_aggregate_different_split_numbers(report):
    result = report.metrics.report_metrics()

    assert_index_equal(result.columns, pd.Index(["mean", "std"]))
    assert len(result) == ((2 * 2) + (4 * 1)) * 2


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


def test_accuracy(report):
    result = report.metrics.report_metrics(
        scoring=["accuracy"],
        aggregate=None,
    )

    assert_index_equal(
        result.columns,
        pd.Index(["DummyClassifier_1", "DummyClassifier_2"], name="Estimator"),
    )
    assert_index_equal(
        result.index,
        pd.MultiIndex.from_tuples(
            [
                ("Accuracy", "Split #0"),
                ("Accuracy", "Split #1"),
                ("Accuracy", "Split #2"),
                ("Accuracy", "Split #3"),
                ("Accuracy", "Split #4"),
            ],
            names=("Metric", "Split"),
        ),
    )


def test_favorability(report):
    result = report.metrics.report_metrics(indicator_favorability=True)

    assert_index_equal(result.columns, pd.Index(["mean", "std", "Favorability"]))
    assert len(result) == 16


def test_regression(report_regression):
    result = report_regression.metrics.report_metrics()

    assert_index_equal(result.columns, pd.Index(["mean", "std"]))
    assert_index_equal(
        result.index,
        pd.MultiIndex.from_tuples(
            [
                ("R²", "DummyRegressor_1"),
                ("RMSE", "DummyRegressor_1"),
                ("Fit time", "DummyRegressor_1"),
                ("Predict time", "DummyRegressor_1"),
                ("R²", "DummyRegressor_2"),
                ("RMSE", "DummyRegressor_2"),
                ("Fit time", "DummyRegressor_2"),
                ("Predict time", "DummyRegressor_2"),
            ],
            names=["Metric", "Estimator"],
        ),
    )


def test_cache(report):
    result = report.metrics.report_metrics()
    cached_result = report.metrics.report_metrics()

    assert_frame_equal(result, cached_result)


def test_init_with_report_names(classification_data):
    """If the estimators are passed as a dict,
    then the estimator names are the dict keys."""

    X, y = classification_data
    cv_report1 = CrossValidationReport(make_classifier(), X, y)
    cv_report2 = CrossValidationReport(make_classifier(), X, y)

    comp = ComparisonReport({"r1": cv_report1, "r2": cv_report2})

    assert_index_equal(
        comp.metrics.report_metrics(aggregate=None).columns,
        pd.Index(["r1", "r2"], name="Estimator"),
    )


def test_report_metrics_X_y(report, classification_data):
    X, y = classification_data
    with pytest.raises(NotImplementedError):
        report.metrics.report_metrics(data_source="X_y", X=X, y=y)
