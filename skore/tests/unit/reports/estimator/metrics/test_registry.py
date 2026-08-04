"""Tests for the metrics registry that are specific to ``EstimatorReport``."""

import pickle
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
)

from skore import EstimatorReport
from skore._utils._testing import check_cache_changed, check_cache_unchanged
from tests.unit.reports._registry_helpers import (
    business_loss_metric,
    business_loss_scorer,
)

business_loss_sklearn_scorer = make_scorer(
    business_loss_metric,
    greater_is_better=False,
    response_method="predict",
    cost_fp=10,
    cost_fn=5,
)


def test_pos_label(binary_classification_report):
    """``pos_label`` from the report flows to a user scorer using it."""
    report = binary_classification_report

    report.metrics.add(
        make_scorer(precision_score, average="binary", pos_label=0),
        name="precision_0",
    )
    display = report.metrics.summarize(metric=["precision_0"])
    assert display.summary["label"].tolist() == [0]


def test_callable_missing_kwargs_hint(binary_classification_report):
    """Error message of the EstimatorReport accessor includes a usage hint."""
    report = binary_classification_report

    err_msg = re.escape(
        "Callable 'business_loss_scorer' has required parameter(s) "
        "('cost_fp', 'cost_fn') not covered by the provided kwargs."
        " Pass those kwargs to add: "
        "add(business_loss_scorer, cost_fp=..., cost_fn=...)"
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.add(business_loss_scorer)


def test_summarize_with_explicit_custom_metric(binary_classification_report):
    """Single-row layout of ``summarize(metric=...)`` for ``EstimatorReport``."""
    report = binary_classification_report

    report.metrics.add(business_loss_sklearn_scorer)

    display = report.metrics.summarize(metric="business_loss_metric")

    assert len(display.summary) == 1
    row = display.summary.iloc[0]
    assert row["verbose_name"] == "Business Loss Metric"
    assert not row["greater_is_better"]


def test_sklearn_scorer_is_cached(binary_classification_report):
    """Test that metric results are cached when metric is a sklearn scorer."""
    report = binary_classification_report

    def my_metric(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    scorer = make_scorer(my_metric, response_method="predict")
    report.metrics.add(scorer)

    with check_cache_changed(report._cache):
        report.metrics.summarize(metric="my_metric")

    with check_cache_unchanged(report._cache):
        report.metrics.summarize(metric="my_metric")

    assert len(report._cache) >= 2


def test_duplicate_add_keeps_existing_cache(binary_classification_report):
    """Duplicate add fails and leaves existing metric cache untouched."""
    report = binary_classification_report

    def metric1(y_true, y_pred):
        return 0.1

    def metric2(y_true, y_pred):
        return 0.2

    scorer1 = make_scorer(metric1, response_method="predict")
    scorer2 = make_scorer(metric2, response_method="predict")

    report.metrics.add(scorer1)
    report.metrics.add(scorer2)

    report.metrics.summarize(metric="metric1")
    report.metrics.summarize(metric="metric2")

    def metric1(y_true, y_pred):
        return 0.3

    scorer1_v2 = make_scorer(metric1, response_method="predict")
    err_msg = re.escape(
        "Cannot add 'metric1': it already exists. Remove it first using the "
        "`remove` method."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.add(scorer1_v2)

    with check_cache_unchanged(report._cache):
        result2 = report.metrics.summarize(metric="metric2")

    with check_cache_unchanged(report._cache):
        result1 = report.metrics.summarize(metric="metric1")

    assert result1.summary["score"].iloc[0] == 0.1
    assert result2.summary["score"].iloc[0] == 0.2


def test_different_metrics_have_separate_cache(binary_classification_report):
    """Test that different metrics don't share cache entries."""
    report = binary_classification_report

    def metric1(y_true, y_pred):
        return 0.1

    def metric2(y_true, y_pred):
        return 0.9

    scorer1 = make_scorer(metric1, response_method="predict")
    scorer2 = make_scorer(metric2, response_method="predict")

    report.metrics.add(scorer1)
    report.metrics.add(scorer2)

    result1 = report.metrics.summarize(metric="metric1")
    result2 = report.metrics.summarize(metric="metric2")

    assert result1.summary["score"].iloc[0] == 0.1
    assert result2.summary["score"].iloc[0] == 0.9


def test_on_report_without_train_data(logistic_binary_classification_with_train_test):
    """Adding still works without train data; summarize on train records an error."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator.fit(X_train, y_train), X_test=X_test, y_test=y_test
    )

    scorer = make_scorer(accuracy_score, response_method="predict")
    report.metrics.add(scorer)

    display = report.metrics.summarize(metric="accuracy_score", data_source="train")
    assert any("No train data were provided" in str(err) for _, err in display.errors)


def test_serde(binary_classification_report):
    """Test that added metrics survive pickle/unpickle with metadata."""
    report = binary_classification_report

    report.metrics.add(
        business_loss_scorer, greater_is_better=False, cost_fp=20, cost_fn=3
    )

    report2 = pickle.loads(pickle.dumps(report))

    assert "business_loss_scorer" in report2._metric_registry

    metric = report2._metric_registry["business_loss_scorer"]
    assert callable(metric.function)
    assert metric.name == "business_loss_scorer"
    assert metric.verbose_name == "Business Loss Scorer"
    assert metric.greater_is_better is False
    assert metric.kwargs == {"cost_fp": 20, "cost_fn": 3}

    display = report2.metrics.summarize()
    assert "Business Loss Scorer" in display.summary["verbose_name"].values


def test_serde_lambda(binary_classification_report):
    """Test that if added metric is a lambda, it is lost when pickling."""
    report = binary_classification_report

    scorer = make_scorer(lambda y_true, y_pred: np.abs(y_true - y_pred).mean())
    report.metrics.add(scorer)
    assert report._metric_registry["<lambda>"].function is not None

    report2 = pickle.loads(pickle.dumps(report))
    assert report2._metric_registry["<lambda>"].function is None

    display = report2.metrics.summarize()
    assert "Metric '<lambda>' has no scoring function." in repr(display.errors)

    report.metrics.summarize()
    report3 = pickle.loads(pickle.dumps(report))
    report3.metrics.summarize()


def test_multimetric_scorer(binary_classification_report):
    """Multimetric scorers are unpacked properly."""
    report = binary_classification_report

    def multimetric_scorer(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=None),
        }

    report.metrics.add(make_scorer(multimetric_scorer, response_method="predict"))

    display = report.metrics.summarize(metric="multimetric_scorer")

    assert list(display.summary["verbose_name"]) == [
        "accuracy",
        "precision",
        "precision",
    ]
    assert list(display.summary["label"]) == [pd.NA, np.int64(0), np.int64(1)]


def test_multimetric_estimator_score(logistic_binary_classification_with_train_test):
    """Setting an estimator's ``score`` method to a multimetric scorer works."""

    def multimetric_scorer(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=None),
        }

    class MyEstimator(LogisticRegression):
        def score(self, X, y, sample_weight=None):
            y_pred = self.predict(X)
            return multimetric_scorer(y, y_pred)

    _, X_train, X_test, y_train, y_test = logistic_binary_classification_with_train_test

    report = EstimatorReport(
        MyEstimator(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.metrics.summarize(metric="score")

    assert list(display.summary["verbose_name"]) == [
        "accuracy",
        "precision",
        "precision",
    ]
    assert list(display.summary["label"]) == [pd.NA, np.int64(0), np.int64(1)]


def test_multimetric_preexisting_metric_name(binary_classification_report):
    """A multimetric scorer submetric can share a built-in verbose name."""
    report = binary_classification_report

    def multimetric_scorer(y_true, y_pred):
        return {"Accuracy": 1000}

    report.metrics.add(make_scorer(multimetric_scorer, response_method="predict"))

    display = report.metrics.summarize()

    assert display.summary["verbose_name"].tolist().count("Accuracy") == 2

    result = display.frame(flat_index=False, verbose_name=True)
    metric_names = result.index.get_level_values("Metric").tolist()
    assert metric_names.count("Accuracy") == 2
