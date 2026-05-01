"""Tests for metrics registry functionality."""

import functools
import pickle
import re

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    get_scorer,
    make_scorer,
    mean_squared_error,
    precision_score,
)

from skore import EstimatorReport
from skore._sklearn.metrics import FunctionKind, Metric, MissingKwargsError
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


def detection_failure_cost(y_true, y_pred_proba, threshold=0.5):
    """Custom metric based on probability threshold."""
    y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
    return business_loss(y_true, y_pred, cost_fp=10, cost_fn=5)


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


@pytest.fixture
def regression_report(linear_regression_with_train_test):
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


class TestBasicAdd:
    """Test basic metric add functionality."""

    def test_sklearn_scorer(self, binary_classification_report):
        """Test adding a sklearn scorer (made with make_scorer)."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)

        metric = report._metric_registry["business_loss"]

        assert metric.name == "business_loss"
        assert metric.verbose_name == "Business Loss"
        assert metric.greater_is_better is False
        assert metric.kwargs == {"cost_fn": 5, "cost_fp": 10}

    def test_callable_missing_kwargs(self, binary_classification_report):
        """Test adding a callable with required params but no kwargs errors."""
        report = binary_classification_report

        err_msg = re.escape(
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
            " Pass those kwargs to add: "
            "add(business_loss_scorer, cost_fp=..., cost_fn=...)"
        )
        with pytest.raises(Exception, match=err_msg):
            report.metrics.add(business_loss_scorer)

    def test_callable_with_name(self, binary_classification_report):
        """Test adding a callable with a custom name."""
        report = binary_classification_report

        report.metrics.add(
            business_loss_scorer, name="custom_metric", cost_fp=10, cost_fn=5
        )

        assert "custom_metric" in report._metric_registry
        assert report._metric_registry["custom_metric"].verbose_name == "Custom Metric"

    def test_callable_with_kwargs(self, binary_classification_report):
        """Test adding a callable with default kwargs via **kwargs."""
        report = binary_classification_report

        report.metrics.add(business_loss_scorer, cost_fp=20, cost_fn=3)

        metric = report._metric_registry["business_loss_scorer"]
        assert metric.kwargs == {"cost_fp": 20, "cost_fn": 3}

        display = report.metrics.summarize(metric="business_loss_scorer")
        assert display.data["score"].notna().all()

    def test_pos_label(self, binary_classification_report):
        """Test adding a scorer with `pos_label` set."""
        report = binary_classification_report

        report.metrics.add(
            make_scorer(precision_score, average="binary", pos_label=0),
            name="precision_0",
        )
        display = report.metrics.summarize(metric=["precision_0"])
        assert display.data["label"].tolist() == [0]

    def test_metric_instance(self, binary_classification_report):
        """Test adding a Metric instance directly."""
        report = binary_classification_report

        metric = Metric.new(get_scorer("accuracy"), name="custom_acc")
        report.metrics.add(metric)

        assert "custom_acc" in report._metric_registry
        display = report.metrics.summarize(metric="custom_acc")
        assert display.data["score"].iloc[0] > 0

    def test_multiple_metrics(self, binary_classification_report):
        """Test adding multiple custom metrics."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)
        report.metrics.add(make_scorer(accuracy_score, response_method="predict"))

        assert "business_loss" in report._metric_registry
        assert "accuracy_score" in report._metric_registry

    def test_cannot_override_builtin_metric(self, binary_classification_report):
        """Test that adding with a built-in technical name raises an error."""
        report = binary_classification_report

        def accuracy(y_true, y_pred):
            return 1.0

        err_msg = "Cannot add 'accuracy': it is a built-in metric name."
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.add(make_scorer(accuracy))


class TestRemove:
    """Test metric remove functionality."""

    def test_remove_custom_metric(self, binary_classification_report):
        """Removing a custom metric drops it from the registry."""
        report = binary_classification_report
        report.metrics.add(custom_scorer)
        assert "business_loss" in report._metric_registry

        report.metrics.remove("business_loss")

        assert "business_loss" not in report._metric_registry

    def test_remove_unknown_metric_raises(self, binary_classification_report):
        """Removing a name that was never added raises KeyError."""
        report = binary_classification_report
        with pytest.raises(KeyError) as exc_info:
            report.metrics.remove("no_such_metric")
        assert exc_info.value.args[0] == "no_such_metric"

    def test_remove_builtin_metric(self, binary_classification_report):
        """Built-in metrics can be removed from the registry."""
        report = binary_classification_report
        assert "accuracy" in report._metric_registry

        report.metrics.remove("accuracy")

        assert "accuracy" not in report._metric_registry
        frame = report.metrics.summarize().frame()
        assert "Accuracy" not in frame.index

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
    """Test integration with the summarize() method."""

    def test_summarize_includes_added_metrics(self, binary_classification_report):
        """Test that summarize() automatically includes added metrics."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)

        display = report.metrics.summarize()

        # Should include both built-in and custom metrics
        assert "Accuracy" in display.frame().index
        assert "Business Loss" in display.frame().index

    def test_summarize_with_explicit_custom_metric(self, binary_classification_report):
        """Test calling summarize with explicit custom metric name."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)

        # Should be able to call by name
        display = report.metrics.summarize(metric="business_loss")

        assert len(display.data) == 1
        row = display.data.iloc[0]
        assert row["metric_verbose_name"] == "Business Loss"
        assert not row["greater_is_better"]

    def test_summarize_with_mixed_metrics(self, binary_classification_report):
        """Test summarize with both built-in and custom metrics."""
        report = binary_classification_report

        report.metrics.add(custom_scorer)

        # Should work with list including both types
        display = report.metrics.summarize(metric=["accuracy", "business_loss"])

        assert set(display.data["metric_verbose_name"]) == {"Accuracy", "Business Loss"}


class TestAddPosition:
    """Tests for metric registry ordering (first vs last)."""

    def test_default_first_lifo_before_builtins(self, binary_classification_report):
        """New metrics default to the front; last added is first among customs."""
        report = binary_classification_report

        def metric_a(y_true, y_pred):
            return 0.0

        def metric_b(y_true, y_pred):
            return 1.0

        report.metrics.add(
            make_scorer(metric_a, response_method="predict"), name="metric_a"
        )
        report.metrics.add(
            make_scorer(metric_b, response_method="predict"), name="metric_b"
        )

        keys = list(report._metric_registry.keys())
        assert keys[0] == "metric_b"
        assert keys[1] == "metric_a"
        assert keys[2] == "accuracy"

        display = report.metrics.summarize()
        assert display.data.iloc[0]["metric_verbose_name"] == "Metric B"

    def test_position_last_appends_in_order(self, binary_classification_report):
        """Last-position adds appear after all built-ins, in insertion order."""
        report = binary_classification_report

        def metric_a(y_true, y_pred):
            return 0.0

        def metric_b(y_true, y_pred):
            return 1.0

        report.metrics.add(
            make_scorer(metric_a, response_method="predict"),
            name="metric_a",
            position="last",
        )
        report.metrics.add(
            make_scorer(metric_b, response_method="predict"),
            name="metric_b",
            position="last",
        )

        keys = list(report._metric_registry.keys())
        assert keys[-2] == "metric_a"
        assert keys[-1] == "metric_b"

    def test_mixed_first_and_last(self, binary_classification_report):
        """First metric at front, last metric at end, built-ins unchanged in between."""
        report = binary_classification_report

        def m_first(y_true, y_pred):
            return 0.0

        def m_last(y_true, y_pred):
            return 1.0

        report.metrics.add(
            make_scorer(m_first, response_method="predict"), name="m_first"
        )
        report.metrics.add(
            make_scorer(m_last, response_method="predict"),
            name="m_last",
            position="last",
        )

        keys = list(report._metric_registry.keys())
        assert keys[0] == "m_first"
        assert keys[1] == "accuracy"
        assert keys[-1] == "m_last"

    def test_readd_raises_without_remove(self, binary_classification_report):
        """Re-adding the same metric name raises and asks to remove first."""
        report = binary_classification_report

        def score_v1(y_true, y_pred):
            return 0.0

        report.metrics.add(
            make_scorer(score_v1, response_method="predict"), name="reorder_me"
        )
        keys_after_first = list(report._metric_registry.keys())
        assert keys_after_first[0] == "reorder_me"

        def score_v2(y_true, y_pred):
            return 1.0

        err_msg = re.escape(
            "Cannot add 'reorder_me': it already exists. "
            "Remove it first using the `remove` method."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.add(
                make_scorer(score_v2, response_method="predict"),
                name="reorder_me",
                position="last",
            )

    def test_metric_registry_add_invalid_position(self, binary_classification_report):
        """MetricRegistry.add rejects unknown position values."""
        report = binary_classification_report
        m = Metric(
            name="only_for_position_test",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
        with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
            report._metric_registry.add(m, position="middle")  # type: ignore[arg-type]


class TestCacheBehavior:
    """Test caching behavior with added metrics."""

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

        # At least the metric value and the model predictions
        assert len(report._cache) >= 2

    def test_callable_predictions_not_cached(self, binary_classification_report):
        """
        Test that model predictions are not cached when metric is a plain callable.
        """
        report = binary_classification_report

        def my_scorer(estimator, X, y_true):
            y_pred = estimator.predict(X)
            return accuracy_score(y_true, y_pred)

        report.metrics.add(my_scorer)

        with check_cache_changed(report._cache):
            report.metrics.summarize(metric="my_scorer")

        # Just the metric value, not the model predictions
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

        # Compute both - should cache
        report.metrics.summarize(metric="metric1")
        report.metrics.summarize(metric="metric2")

        # Attempt duplicate add of metric1 with a new function
        def metric1(y_true, y_pred):
            return 0.3

        scorer1_v2 = make_scorer(metric1, response_method="predict")
        err_msg = re.escape(
            "Cannot add 'metric1': it already exists. Remove it first using the "
            "`remove` method."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.add(scorer1_v2)

        # metric2 should use cache
        with check_cache_unchanged(report._cache):
            result2 = report.metrics.summarize(metric="metric2")

        # metric1 should still use cache from the original scorer
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
    """Test edge cases and error conditions."""

    def test_on_report_without_train_data(
        self, logistic_binary_classification_with_train_test
    ):
        """Test that add works even without train data."""
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

    def test_duplicate_name_raises(self, binary_classification_report):
        """Adding with duplicate name raises and keeps the original metric."""
        report = binary_classification_report

        def score(y_true, y_pred):
            return 0

        report.metrics.add(make_scorer(score, response_method="predict"))

        nb_metrics_before_overwriting = len(report._metric_registry)

        result = report.metrics.summarize(metric="score")
        assert result.data["score"].iloc[0] == 0

        # add a new metric with the same name
        def score(y_true, y_pred):
            return 1

        err_msg = re.escape(
            "Cannot add 'score': it already exists. "
            "Remove it first using the `remove` method."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.add(make_scorer(score, response_method="predict"))

        assert len(report._metric_registry) == nb_metrics_before_overwriting

        # summarize still reflects the original metric
        result = report.metrics.summarize(metric="score")
        assert result.data["score"].iloc[0] == 0


class TestDifferentMLTasks:
    """Test that registry works across different ML tasks."""

    def test_multiclass(self, logistic_multiclass_classification_with_train_test):
        """Test add of multiclass-compatible metric."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_multiclass_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        report.metrics.add(make_scorer(accuracy_score, response_method="predict"))

        display = report.metrics.summarize()
        assert "Accuracy Score" in display.data["metric_verbose_name"].values

    def test_regression(self, regression_report):
        """Test add on regression report."""
        report = regression_report

        def custom_mse(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)

        scorer = make_scorer(
            custom_mse,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.add(scorer)

        display = report.metrics.summarize()
        assert "Custom Mse" in display.data["metric_verbose_name"].values

    def test_multioutput_regression(
        self, linear_regression_multioutput_with_train_test
    ):
        """Test add on multioutput regression."""
        estimator, X_train, X_test, y_train, y_test = (
            linear_regression_multioutput_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.add(scorer)

        display = report.metrics.summarize()
        assert "Mean Squared Error" in display.data["metric_verbose_name"].values

    def test_wrong_ml_task(self, linear_regression_with_train_test):
        """adding a metric incompatible with the ML task doesn't crash."""
        estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
        report = EstimatorReport(
            estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        scorer = make_scorer(
            mean_squared_error, greater_is_better=False, response_method="predict"
        )
        report.metrics.add(scorer)


class TestDictReturnValues:
    """Test that metrics returning dicts work correctly (per-label scores).

    Note: Multimetric scorers (single scorer returning multiple different metrics)
    are NOT supported - users should add metrics separately.
    """

    def test_per_class_accuracy_dict(self, binary_classification_report):
        """Test metric that returns per-class scores as dict."""
        report = binary_classification_report

        def per_class_accuracy(y_true, y_pred) -> dict[int, float]:
            """Return accuracy for each class."""
            accuracies = {}
            for label in np.unique(y_true):
                mask = y_true == label
                accuracies[int(label)] = float((y_pred[mask] == label).mean())
            return accuracies

        def scorer(est, X, y_true):
            y_pred = est.predict(X)
            return per_class_accuracy(y_true, y_pred)

        report.metrics.add(scorer, name="per_class_accuracy")

        display = report.metrics.summarize(metric="per_class_accuracy")

        metric_rows = display.data[
            display.data["metric_verbose_name"] == "Per Class Accuracy"
        ]
        assert len(metric_rows) == 2
        assert set(metric_rows["label"].values) == {0, 1}

        # Cached correctly
        with check_cache_unchanged(report._cache):
            report.metrics.summarize(metric="per_class_accuracy")

    def test_multimetric_scorer_not_recommended(self, binary_classification_report):
        """Multimetric scorers are treated as per-label scores (not supported)."""
        report = binary_classification_report

        def multimetric_scorer(y_true, y_pred):
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary"),
            }

        report.metrics.add(make_scorer(multimetric_scorer, response_method="predict"))

        display = report.metrics.summarize(metric="multimetric_scorer")

        metric_rows = display.data[
            display.data["metric_verbose_name"] == "Multimetric Scorer"
        ]
        # Not quite right...
        assert set(metric_rows["label"]) == {"accuracy", "precision"}


class TestStringScorerNames:
    """Test support for string scorer names (via sklearn.metrics.get_scorer)."""

    def test_string_scorer_name(self, binary_classification_report):
        """Test adding a metric using its sklearn string name."""
        report = binary_classification_report

        report.metrics.add("f1")
        assert "f1" in report._metric_registry

        # NOTE: User can pass "f1", not "f1_score" which is the name of the actual
        # metric function
        display = report.metrics.summarize(metric="f1")
        metric_rows = display.data[display.data["metric_verbose_name"] == "F1"]

        assert len(metric_rows) == 1

    def test_string_scorer_appears_in_summarize(self, binary_classification_report):
        """Test that string scorers appear in summarize() output."""
        report = binary_classification_report

        display = report.metrics.summarize()
        metrics_before = set(display.data["metric_verbose_name"])

        report.metrics.add("f1")

        display = report.metrics.summarize()
        metrics_after = set(display.data["metric_verbose_name"])

        assert metrics_after - metrics_before == {"F1"}

    def test_neg_scorer(self, regression_report):
        """Test that neg_* scorers have correct sign, direction, and display name."""
        report = regression_report

        report.metrics.add(get_scorer("neg_mean_squared_error"))

        # `neg_` was stripped off metric name
        assert "mean_squared_error" in report._metric_registry

        display = report.metrics.summarize(metric="mean_squared_error")
        row = display.data.iloc[0]

        assert row["score"] >= 0
        assert not row["greater_is_better"]
        assert not row["metric_verbose_name"].lower().startswith("neg")

    def test_without_neg_prefix(self, regression_report):
        """Test that metric strings passed without 'neg_' prefix can be added and
        duplicate registration raises an explicit error."""
        report = regression_report

        report.metrics.add("mean_squared_error")
        assert "mean_squared_error" in report._metric_registry

        err_msg = re.escape(
            "Cannot add 'mean_squared_error': it already exists. "
            "Remove it first using the `remove` method."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.add("neg_mean_squared_error")

    def test_invalid_string_scorer_name(self, binary_classification_report):
        """Test that invalid sklearn scorer names raise an error."""
        report = binary_classification_report

        with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
            report.metrics.add("xyz")


class TestMetric:
    def test_repr(self):
        """Test that Metric.__repr__ works as expected"""
        m = Metric(name="accuracy", function=None, greater_is_better=True)
        assert repr(m) == (
            "Metric(name='accuracy', verbose_name='Accuracy', function=None, "
            "greater_is_better=True, response_method=None, kwargs={})"
        )

    def test_repr_kwargs(self):
        """Test that Metric.__repr__ works as expected when kwargs are passed."""
        m = Metric(
            name="accuracy",
            function=None,
            greater_is_better=True,
            kwargs={"hello": 1},
        )

        assert repr(m) == (
            "Metric(name='accuracy', verbose_name='Accuracy', function=None, "
            "greater_is_better=True, response_method=None, kwargs={'hello': 1})"
        )

    def test_greater_is_better_none(self):
        """Test that the metric stores an undefined direction when applicable."""
        m = Metric(name="test", greater_is_better=None)
        assert m.greater_is_better is None

        m = Metric(name="test", greater_is_better=True)
        assert m.greater_is_better is True


class TestSerialization:
    """Test that added metrics survive pickling (for Project storage)."""

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

        # if we cache beforehand, then it works:
        report.metrics.summarize()
        report3 = pickle.loads(pickle.dumps(report))
        report3.metrics.summarize()


class TestMetricNew:
    """Test the Metric.new method."""

    def test_callable(self):
        """Test creating a Metric from a callable."""
        metric = Metric.new(
            business_loss_scorer,
            greater_is_better=False,
            kwargs={"cost_fp": 10, "cost_fn": 5},
        )

        assert isinstance(metric, Metric)
        assert metric.name == "business_loss_scorer"
        assert metric.function is business_loss_scorer
        assert metric.greater_is_better is False
        assert metric.kwargs == {"cost_fp": 10, "cost_fn": 5}

    def test_callable_with_name(self):
        """Test creating a Metric from a callable with a custom name."""
        metric = Metric.new(
            business_loss_scorer, name="my_loss", kwargs={"cost_fp": 10, "cost_fn": 5}
        )

        assert metric.name == "my_loss"
        assert metric.verbose_name == "My Loss"
        assert metric.function is business_loss_scorer
        assert metric.kwargs == {"cost_fp": 10, "cost_fn": 5}

    def test_callable_missing_kwargs(self):
        """Test that Metric.new raises for required params without kwargs."""
        err_msg = re.escape(
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
        )
        with pytest.raises(MissingKwargsError, match=err_msg):
            Metric.new(business_loss_scorer)

    def test_callable_metric_y(self):
        """Test that Metric.new raises for callable metrics taking `y_true` as first
        argument."""
        err_msg = re.escape(
            "Expected a scorer callable with an estimator as its first argument; "
            "got first argument 'y_true'"
        )
        with pytest.raises(TypeError, match=err_msg):
            Metric.new(business_loss)

    def test_callable_metric_not_enough_positional_args(self):
        """Test that Metric.new raises for callable metrics which do not take enough
        positional parameters."""

        # First argument does not start with `y`
        def metric(true_labels, predicted_labels, *, some_kwarg):
            pass

        err_msg = re.escape(
            "Expected a scorer callable with at least 3 positional arguments "
            "(estimator, X, y); got ['true_labels', 'predicted_labels']"
        )
        with pytest.raises(TypeError, match=err_msg):
            Metric.new(metric)

    def test_scorer(self):
        """Test creating a Metric from an sklearn scorer."""
        metric = Metric.new(
            business_loss_scorer,
            greater_is_better=False,
            kwargs={"cost_fp": 10, "cost_fn": 5},
        )

        assert isinstance(metric, Metric)
        assert metric.name == "business_loss_scorer"
        assert metric.function is business_loss_scorer
        assert metric.greater_is_better is False
        assert metric.kwargs == {"cost_fp": 10, "cost_fn": 5}

    def test_metric(self):
        """Test creating a Metric from an existing Metric."""
        original = Metric(name="original", function=get_scorer("accuracy"))
        result = Metric.new(original)

        assert result.name == "original"
        assert result is original

    def test_metric_with_name(self):
        """Test creating a Metric from a Metric with name override."""
        original = Metric(name="original", function=get_scorer("accuracy"))
        result = Metric.new(original, name="renamed")

        assert result.name == "renamed"
        assert result.verbose_name == "Renamed"
        assert original.name == "original"  # unchanged

    def test_string(self):
        """Test creating a Metric from an sklearn scorer string name."""
        metric = Metric.new("f1")

        assert isinstance(metric, Metric)
        assert metric.name == "f1"  # not "f1_score"
        assert metric.function is not None

    def test_invalid_string(self):
        """Test that an invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            Metric.new("xyz")

    def test_invalid_type(self):
        """Test that passing an invalid type raises an error."""
        with pytest.raises(TypeError, match="Cannot create"):
            Metric.new(42)

    def test_functools_partial(self):
        """Test creating a Metric from a functools.partial."""
        partial_func = functools.partial(business_loss_scorer, cost_fp=10, cost_fn=5)
        metric = Metric.new(partial_func)

        assert metric.name == "business_loss_scorer"
        assert metric.function is partial_func

    def test_callable_object_without_name(self):
        """Test creating a Metric from a callable without __name__."""

        class MyScorer:
            def __call__(self, estimator, X, y):
                return get_scorer("accuracy")(estimator, X, y)

        metric = Metric.new(MyScorer())

        assert metric.name == "MyScorer"


def test_available_default(binary_classification_report):
    """Test that the default available() returns True."""
    m = Metric(name="test")
    assert m.available(binary_classification_report) is True


def test_call_no_function(binary_classification_report):
    """Test that calling a Metric with no function raises."""
    m = Metric(name="abstract_metric", function=None)
    err_msg = "Metric 'abstract_metric' has no scoring function."
    with pytest.raises(ValueError, match=err_msg):
        m(report=binary_classification_report)


def test_metric_registry_repr(binary_classification_report):
    """Test the MetricRegistry repr."""
    registry = binary_classification_report._metric_registry
    result = repr(registry)
    assert result.startswith("MetricRegistry")
    assert "accuracy" in result
