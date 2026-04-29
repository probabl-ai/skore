"""Tests for the metrics registry that are specific to ``EstimatorReport``.

Behaviors that hold across every report kind live in
``tests/unit/reports/test_metrics_registry_common.py``. This module keeps only
tests that exercise EstimatorReport-specific surfaces such as the top-level
``report._cache``, the ``pos_label`` constructor argument, the no-train-data
flow, the EstimatorReport-specific accessor error message, ``summarize`` row
count, and pickle serialization.
"""

import pickle
import re

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
)

from skore import EstimatorReport
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def business_loss(y_true, y_pred, *, cost_fp, cost_fn):
    """Custom business metric: weighted cost of false positives and negatives."""
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn


def business_loss_scorer(estimator, X, y, cost_fp, cost_fn):
    y_pred = estimator.predict(X)
    return business_loss(y, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)


custom_scorer = make_scorer(
    business_loss,
    greater_is_better=False,
    response_method="predict",
    cost_fp=10,
    cost_fn=5,
)


@pytest.fixture
def binary_classification_report(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label=1,
    )


class TestBasicAdd:
    """EstimatorReport-specific behaviors of ``metrics.add``."""

    def test_pos_label(self, binary_classification_report):
        """``pos_label`` from the report flows to a user scorer using it."""
        report = binary_classification_report

        report.metrics.add(
            make_scorer(precision_score, average="binary", pos_label=0),
            name="precision_0",
        )
        display = report.metrics.summarize(metric=["precision_0"])
        assert display.data["label"].tolist() == [0]

    def test_callable_missing_kwargs(self, binary_classification_report):
        """Error message of the EstimatorReport accessor includes a usage hint."""
        report = binary_classification_report

        err_msg = re.escape(
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
            " Pass those kwargs to add: "
            "add(business_loss_scorer, cost_fp=..., cost_fn=...)"
        )
        with pytest.raises(Exception, match=err_msg):
            report.metrics.add(business_loss_scorer)


class TestRemove:
    def test_remove_invalidates_cache_only_for_removed_metric(
        self, binary_classification_report
    ):
        """Removing a metric clears its cache entries only."""
        report = binary_classification_report

        def metric1(y_true, y_pred):
            return 0.1

        def metric2(y_true, y_pred):
            return 0.2

        report.metrics.add(make_scorer(metric1, response_method="predict"))
        report.metrics.add(make_scorer(metric2, response_method="predict"))
        report.metrics.summarize(metric="metric1")
        report.metrics.summarize(metric="metric2")

        report.metrics.remove("metric1")

        assert not any(k[1] == "metric1" for k in report._cache)
        assert any(k[1] == "metric2" for k in report._cache)


class TestSummarizeIntegration:
    def test_summarize_with_explicit_custom_metric(self, binary_classification_report):
        """Single-row layout of ``summarize(metric=...)`` for ``EstimatorReport``."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)

        display = report.metrics.summarize(metric="business_loss")

        assert len(display.data) == 1
        row = display.data.iloc[0]
        assert row["metric_verbose_name"] == "Business Loss"
        assert not row["greater_is_better"]


class TestCacheBehavior:
    """``EstimatorReport._cache`` is the only top-level cache among reports."""

    def test_sklearn_scorer_is_cached(self, binary_classification_report):
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

    def test_callable_predictions_not_cached(self, binary_classification_report):
        """Predictions are not cached when the metric is a plain callable."""
        report = binary_classification_report

        def my_scorer(estimator, X, y_true):
            y_pred = estimator.predict(X)
            return accuracy_score(y_true, y_pred)

        report.metrics.add(my_scorer)

        with check_cache_changed(report._cache):
            report.metrics.summarize(metric="my_scorer")

        assert len(report._cache) == 1

    def test_duplicate_add_keeps_existing_cache(self, binary_classification_report):
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

        def metric1(y_true, y_pred):  # noqa: F811
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

        assert result1.data["score"].iloc[0] == 0.1
        assert result2.data["score"].iloc[0] == 0.2

    def test_different_metrics_have_separate_cache(self, binary_classification_report):
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

        assert result1.data["score"].iloc[0] == 0.1
        assert result2.data["score"].iloc[0] == 0.9


class TestEdgeCases:
    """No-train-data flow is unique to ``EstimatorReport``."""

    def test_on_report_without_train_data(
        self, logistic_binary_classification_with_train_test
    ):
        """Adding still works without train data; summarize on train fails clearly."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(accuracy_score, response_method="predict")
        report.metrics.add(scorer)

        with pytest.raises(ValueError, match="(?i)train|data"):
            report.metrics.summarize(metric="accuracy_score", data_source="train")


class TestSerialization:
    """Pickling round-trips a single ``EstimatorReport`` and its registry."""

    def test_serde(self, binary_classification_report):
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
        assert "Business Loss Scorer" in display.data["metric_verbose_name"].values

    def test_serde_lambda(self, binary_classification_report):
        """Test that if added metric is a lambda, it is lost when pickling."""
        report = binary_classification_report

        scorer = make_scorer(lambda y_true, y_pred: np.abs(y_true - y_pred).mean())
        report.metrics.add(scorer)
        assert report._metric_registry["<lambda>"].function is not None

        report2 = pickle.loads(pickle.dumps(report))
        assert report2._metric_registry["<lambda>"].function is None

        err_msg = "Metric '<lambda>' has no scoring function."
        with pytest.raises(ValueError, match=err_msg):
            report2.metrics.summarize()

        report.metrics.summarize()
        report3 = pickle.loads(pickle.dumps(report))
        report3.metrics.summarize()
