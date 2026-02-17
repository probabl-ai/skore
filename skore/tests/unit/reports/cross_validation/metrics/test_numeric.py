import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.svm import SVC

from skore import CrossValidationReport


def _normalize_metric_name(index):
    """Helper to normalize the metric name present in a pandas index that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = index[0] if isinstance(index, tuple) else index
    # Remove spaces and underscores
    return re.sub(r"[^a-zA-Z]", "", s.lower())


def _check_metrics_names(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for idx in result.index:
        normalized_idx = _normalize_metric_name(idx)
        matches = [metric for metric in normalized_expected if metric == normalized_idx]
        assert len(matches) == 1, (
            f"No match found for index '{idx}' in expected metrics:  {expected_metrics}"
        )


def _check_results_single_metric(report, metric, expected_n_splits, expected_nb_stats):
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)(aggregate=None)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == expected_n_splits
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)(aggregate=None)
    pd.testing.assert_frame_equal(result, result_with_cache)

    # check that the columns contains the expected split names
    split_names = result.columns.get_level_values(1).unique()
    expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
    assert list(split_names) == expected_split_names

    # check that something was written to the cache
    assert report._cache != {}
    report.clear_cache()

    _check_metrics_names(result, [metric], expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = getattr(report.metrics, metric)(aggregate=stats)
    # check that the columns contains the expected split names
    split_names = result.columns.get_level_values(1).unique()
    assert list(split_names) == stats

    stats = "mean"
    result = getattr(report.metrics, metric)(aggregate=stats)
    # check that the columns contains the expected split names
    split_names = result.columns.get_level_values(1).unique()
    assert list(split_names) == [stats]


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("accuracy", 1),
        ("precision", 2),
        ("recall", 2),
        ("brier_score", 1),
        ("roc_auc", 1),
        ("log_loss", 1),
    ],
)
def test_binary_classification(forest_binary_classification_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    (estimator, X, y), cv = forest_binary_classification_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("accuracy", 1),
        ("precision", 3),
        ("recall", 3),
        ("roc_auc", 3),
        ("log_loss", 1),
    ],
)
def test_multiclass_classification(
    forest_multiclass_classification_data, metric, nb_stats
):
    """Check the behaviour of the metrics methods available for multiclass
    classification.
    """
    (estimator, X, y), cv = forest_multiclass_classification_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize("metric, nb_stats", [("r2", 1), ("rmse", 1)])
def test_regression(linear_regression_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = linear_regression_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize("metric, nb_stats", [("r2", 2), ("rmse", 2)])
def test_regression_multioutput(linear_regression_multioutput_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = linear_regression_multioutput_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


def test_brier_score_requires_probabilities():
    """Check that the Brier score is not defined for estimator that do not
    implement `predict_proba`.

    Non-regression test for:
    https://github.com/probabl-ai/skore/pull/1471
    """
    estimator = SVC()  # SVC does not implement `predict_proba` with default parameters
    X, y = make_classification(n_classes=2, random_state=42)

    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    assert not hasattr(report.metrics, "brier_score")


def test_custom_metric(forest_binary_classification_data):
    """Check that we can compute a custom metric."""
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    result = report.metrics.custom_metric(
        metric_function=accuracy_score,
        response_method="predict",
    )
    assert result.shape == (1, 2)
    assert result.index == ["Accuracy Score"]


def test_cache_key_with_string_aggregate_is_not_split(
    forest_binary_classification_data,
):
    """
    Check that string aggregate values are stored as a single cache-key item.
    Non-regression test for: https://github.com/probabl-ai/skore/issues/2450
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    report.metrics.summarize(aggregate="mean")

    summarize_cache_keys = [key for key in report._cache if key[1] == "summarize"]
    assert summarize_cache_keys
    assert any("mean" in key for key in summarize_cache_keys)


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_summarize_pos_label_overwrite(metric, logistic_binary_classification_data):
    """Check that `pos_label` can be overwritten in `summarize`"""
    classifier, X, y = logistic_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report = CrossValidationReport(classifier, X, y)
    result_both_labels = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])

    report = CrossValidationReport(classifier, X, y, pos_label="B")
    result = report.metrics.summarize(metric=metric).frame().reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "B"), (report.estimator_name_, "mean")
        ]
    )

    result = (
        report.metrics.summarize(metric=metric, pos_label="A").frame().reset_index()
    )
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "A"), (report.estimator_name_, "mean")
        ]
    )


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_overwrite(
    metric, logistic_binary_classification_data
):
    """Check that `pos_label` can be overwritten in `summarize`."""
    classifier, X, y = logistic_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report = CrossValidationReport(classifier, X, y)
    result_both_labels = getattr(report.metrics, metric)().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])

    result = getattr(report.metrics, metric)(pos_label="B").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "B"), (report.estimator_name_, "mean")
        ]
    )

    result = getattr(report.metrics, metric)(pos_label="A").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    assert (
        result.loc[metric.capitalize(), (report.estimator_name_, "mean")]
        == result_both_labels.loc[
            (metric.capitalize(), "A"), (report.estimator_name_, "mean")
        ]
    )


def test_invalid_X_y_call_still_raises_after_cache_write(
    logistic_binary_classification_data,
):
    """
    Non regression for
    Invalid `X`/`y` args should not be masked by a cache hit.
    """
    classifier, X, y = logistic_binary_classification_data
    report = CrossValidationReport(classifier, X, y)

    error_msg = "X and y must be None when data_source is test"
    with pytest.raises(ValueError, match=error_msg):
        report.metrics.accuracy(X=X, y=y)

    report.metrics.accuracy()

    with pytest.raises(ValueError, match=error_msg):
        report.metrics.accuracy(X=X, y=y)
