import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
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
        matches = [
            metric
            for metric in normalized_expected
            if normalized_idx == _normalize_metric_name(metric)
            or normalized_idx.startswith(_normalize_metric_name(metric))
        ]
        assert matches, (
            f"No match found for index '{idx}' in expected metrics:  {expected_metrics}"
        )


def _stat_suffixes(columns):
    if isinstance(columns, pd.MultiIndex):
        return list(columns.get_level_values(1))
    return [col.rsplit("_", 1)[-1] for col in columns]


def _split_indices(columns):
    if isinstance(columns, pd.MultiIndex):
        return sorted(
            int(str(col[1]).replace("Split #", ""))
            for col in columns
            if str(col[1]).startswith("Split #")
        )
    return sorted(
        int(col.rsplit("_split_", 1)[1]) for col in columns if "_split_" in col
    )


def _check_results_single_metric(report, metric, expected_n_splits, expected_nb_stats):
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)(aggregate=None)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == expected_n_splits
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)(aggregate=None)
    pd.testing.assert_frame_equal(result, result_with_cache)

    assert _split_indices(result.columns) == list(range(expected_n_splits))

    # check that something was written to the children's cache
    assert all(report._cache != {} for report in report.reports_)
    report._clear_cache()

    _check_metrics_names(result, [metric], expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = getattr(report.metrics, metric)(aggregate=stats)
    assert _stat_suffixes(result.columns) == stats

    stats = "mean"
    result = getattr(report.metrics, metric)(aggregate=stats)
    assert _stat_suffixes(result.columns) == [stats]


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("accuracy", 1),
        ("precision", 2),
        ("recall", 2),
        ("brier_score", 1),
        ("roc_auc", 1),
        ("log_loss", 1),
        ("score", 1),
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
        ("precision", 4),
        ("recall", 4),
        ("roc_auc", 4),
        ("log_loss", 1),
        ("score", 1),
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


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("r2", 1),
        ("rmse", 1),
        ("mae", 1),
        ("mape", 1),
        ("score", 1),
    ],
)
def test_regression(linear_regression_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = linear_regression_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("r2", 2),
        ("rmse", 2),
        ("mae", 2),
        ("mape", 2),
        ("score", 1),
    ],
)
def test_regression_multioutput(linear_regression_multioutput_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = linear_regression_multioutput_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize("metric", ["mae", "mape"])
def test_regression_multioutput_array_weights(
    linear_regression_multioutput_data, metric
):
    """Check that mae and mape accept an array of weights for multioutput."""
    (estimator, X, y), cv = linear_regression_multioutput_data, 2
    report = CrossValidationReport(estimator, X, y, splitter=cv)
    weights = np.array([0.3, 0.7])
    result = getattr(report.metrics, metric)(multioutput=weights)
    assert isinstance(result, pd.DataFrame)
    # weighted average produces a single scalar per split, so 1 row
    assert result.shape[0] == 1


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


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_overwrite(
    metric, logistic_binary_classification_data
):
    """Check that `pos_label` can be set."""
    classifier, X, y = logistic_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report = CrossValidationReport(classifier, X, y)
    result_both_labels = getattr(report.metrics, metric)()
    mean_col = next(
        col
        for col in result_both_labels.columns
        if (col[1] if isinstance(col, tuple) else col).endswith("mean")
        or (isinstance(col, tuple) and col[1] == "mean")
    )
    label_rows = {
        str(label).lower(): index
        for index in result_both_labels.index
        for label in [index[1] if isinstance(index, tuple) else index]
    }

    report = CrossValidationReport(classifier, X, y, pos_label="B")
    result = getattr(report.metrics, metric)()
    assert result.shape[0] == 1
    assert result.iloc[0][mean_col] == result_both_labels.loc[label_rows["b"], mean_col]

    report = CrossValidationReport(classifier, X, y, pos_label="A")
    result = getattr(report.metrics, metric)()
    assert result.shape[0] == 1
    assert result.iloc[0][mean_col] == result_both_labels.loc[label_rows["a"], mean_col]


# report.metrics.get


def test_get(binary_classification_data):
    """``get`` works."""
    X, y = binary_classification_data
    report = CrossValidationReport(
        DummyClassifier(strategy="uniform"), X, y, splitter=2
    )

    assert isinstance(report.metrics.get("precision"), pd.DataFrame)
    with pytest.raises(KeyError):
        report.metrics.get("non-existing metric")


def test_get_custom(binary_classification_data):
    """``get`` works for custom metrics."""
    X, y = binary_classification_data
    report = CrossValidationReport(
        DummyClassifier(strategy="uniform"), X, y, splitter=2
    )

    with pytest.raises(KeyError):
        report.metrics.get("hello")

    report.metrics.add(lambda estimator, X, y: 1, name="hello")

    assert report.metrics.get("hello").to_dict() == {
        ("DummyClassifier", "mean"): {"Hello": 1.0},
        ("DummyClassifier", "std"): {"Hello": 0.0},
    }


def test_custom_metric_as_method(binary_classification_data):
    """Custom metrics are accessible as methods."""
    X, y = binary_classification_data
    report = CrossValidationReport(
        DummyClassifier(strategy="uniform"), X, y, splitter=2
    )

    with pytest.raises(AttributeError):
        report.metrics.hello()

    report.metrics.add(lambda estimator, X, y: 1, name="hello")

    assert report.metrics.hello().to_dict() == {
        ("DummyClassifier", "mean"): {"Hello": 1.0},
        ("DummyClassifier", "std"): {"Hello": 0.0},
    }

    report.metrics.remove("hello")

    with pytest.raises(AttributeError):
        report.metrics.hello()
