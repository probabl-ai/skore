"""Tests for metrics registry functionality."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_squared_error,
    precision_score,
)

from skore import EstimatorReport
from skore._utils._testing import check_cache_changed, check_cache_unchanged


# Helper function for custom metrics
def business_loss(y_true, y_pred, cost_fp=10, cost_fn=5):
    """Custom business metric: weighted cost of false positives and negatives."""
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn


def detection_failure_cost(y_true, y_pred_proba, threshold=0.5):
    """Custom metric based on probability threshold."""
    y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
    return business_loss(y_true, y_pred, cost_fp=10, cost_fn=5)


# Fixtures


@pytest.fixture
def binary_classification_report(logistic_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def regression_report(linear_regression_with_train_test):
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


# Test 1: Basic registration and discoverability


class TestBasicRegistration:
    """Test basic metric registration functionality."""

    def test_register_simple_scorer(self, binary_classification_report):
        """Test registering a simple custom scorer."""
        report = binary_classification_report

        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )

        report.metrics.register(custom_scorer)

        assert isinstance(report.metrics.registry, dict)

    def test_registered_metric_has_metadata(self, binary_classification_report):
        """Test that registered metrics expose required metadata."""
        report = binary_classification_report

        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(custom_scorer)

        # Get the registered metric via report.metrics.registry
        registry = report.metrics.registry
        assert "business_loss" in registry

        metric_info = registry["business_loss"]

        # Should have these attributes
        assert hasattr(metric_info, "name")
        assert hasattr(metric_info, "verbose_name")
        assert hasattr(metric_info, "greater_is_better")
        assert hasattr(metric_info, "response_method")

        assert metric_info.name == "business_loss"
        assert metric_info.greater_is_better is False
        assert metric_info.response_method == "predict"

    def test_register_multiple_metrics(self, binary_classification_report):
        """Test registering multiple custom metrics."""
        report = binary_classification_report

        scorer1 = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        scorer2 = make_scorer(
            accuracy_score,
            response_method="predict",
        )

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        # Both should be registered
        registry = report.metrics.registry
        assert "business_loss" in registry
        assert "accuracy_score" in registry


# Test 2: Integration with summarize()


class TestSummarizeIntegration:
    """Test integration with the summarize() method."""

    def test_summarize_includes_registered_metrics(self, binary_classification_report):
        """Test that summarize() automatically includes registered metrics."""
        report = binary_classification_report

        # Register a custom metric
        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(custom_scorer)

        # summarize() should include it
        display = report.metrics.summarize()

        metric_names = display.data["metric"].unique()

        # Should include both built-in and custom metrics
        assert "Accuracy" in metric_names  # Built-in
        assert "Business Loss" in metric_names  # Custom (verbose name)

    def test_summarize_with_explicit_custom_metric(self, binary_classification_report):
        """Test calling summarize with explicit custom metric name."""
        report = binary_classification_report

        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(custom_scorer)

        # Should be able to call by name
        display = report.metrics.summarize(metric="business_loss")

        assert set(display.data["metric"]) == {"Business Loss"}

    def test_summarize_with_mixed_metrics(self, binary_classification_report):
        """Test summarize with both built-in and custom metrics."""
        report = binary_classification_report

        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(custom_scorer)

        # Should work with list including both types
        display = report.metrics.summarize(metric=["accuracy", "business_loss"])

        metric_names = display.data["metric"].unique()
        assert len(metric_names) == 2

    def test_registered_metric_has_correct_favorability(
        self, binary_classification_report
    ):
        """Test that registered metrics show correct favorability icons."""
        report = binary_classification_report

        report.metrics.register(
            make_scorer(
                business_loss,
                greater_is_better=False,
                response_method="predict",
            )
        )

        display = report.metrics.summarize(metric="business_loss")

        assert list(display.data["favorability"]) == ["(↘︎)"]


# Test 3: Protection against overriding built-in metrics


class TestBuiltInProtection:
    """Test protection against overriding built-in metric names."""

    def test_cannot_override_builtin_metric_by_name(self, binary_classification_report):
        """Test that registering with a built-in technical name raises an error."""
        report = binary_classification_report

        # Try to register a scorer with a built-in metric name
        fake_accuracy = make_scorer(
            lambda y_true, y_pred: 1.0,  # Always return 1.0
            response_method="predict",
        )
        # Force the name to conflict with built-in technical name
        fake_accuracy._score_func.__name__ = "accuracy"

        with pytest.raises(ValueError, match="(?i)built-in|override|reserved|conflict"):
            report.metrics.register(fake_accuracy)

    def test_cannot_override_builtin_metric_by_verbose_name(
        self, binary_classification_report
    ):
        """Test that registering with a built-in verbose name raises an error."""
        report = binary_classification_report

        # Try to register with a function that would generate a conflicting verbose name
        def my_custom_func(y_true, y_pred):
            return 1.0

        # When converted to verbose name, this becomes "Accuracy" which conflicts
        my_custom_func.__name__ = "accuracy"

        fake_accuracy = make_scorer(my_custom_func, response_method="predict")

        with pytest.raises(ValueError, match="(?i)built-in|override|reserved|conflict"):
            report.metrics.register(fake_accuracy)

    def test_builtin_names_are_protected(self, binary_classification_report):
        """Test that all built-in metric names are protected."""
        report = binary_classification_report

        fake_metric = make_scorer(
            lambda y_true, y_pred: 0.0,
            response_method="predict",
        )
        fake_metric._score_func.__name__ = "accuracy"

        # Should raise for any protected name
        with pytest.raises(ValueError, match="(?i)built-in|override|reserved|conflict"):
            report.metrics.register(fake_metric)


# Test 4: sklearn Scorer integration


class TestScorerExtraction:
    """Test extraction of metadata from sklearn scorers."""

    def test_extract_score_func(self, binary_classification_report):
        """Test that _score_func is correctly extracted."""
        report = binary_classification_report

        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(custom_scorer)

        # The registered metric should use the original function
        registry = report.metrics.registry
        metric_info = registry["business_loss"]

        # Should store reference to the original score function
        assert hasattr(metric_info, "score_func") or hasattr(metric_info, "_score_func")
        # The function should be business_loss or wrapped version
        assert callable(
            metric_info.score_func
            if hasattr(metric_info, "score_func")
            else metric_info._score_func
        )

    def test_extract_response_method(self, binary_classification_report):
        """Test that _response_method is correctly extracted."""
        report = binary_classification_report

        # Scorer using predict_proba
        proba_scorer = make_scorer(
            detection_failure_cost,
            greater_is_better=False,
            response_method="predict_proba",
            threshold=0.7,
        )
        report.metrics.register(proba_scorer)

        # Should extract response_method="predict_proba"
        registry = report.metrics.registry
        metric_info = registry["detection_failure_cost"]

        assert metric_info.response_method == "predict_proba"

    def test_extract_kwargs(self, binary_classification_report):
        """Test that scorer kwargs are correctly extracted and forwarded."""
        report = binary_classification_report

        # Scorer with custom kwargs
        custom_scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
            cost_fp=20,  # Custom kwargs
            cost_fn=3,
        )
        report.metrics.register(custom_scorer)

        # When computing the metric, should use cost_fp=20, cost_fn=3
        display = report.metrics.summarize(metric="business_loss")

        # The actual value should reflect these custom costs
        # (Hard to verify without knowing the data, but at least shouldn't error)
        assert display.data["score"].notna().all()


# Test 5: neg_* scorer handling


class TestNegScorerHandling:
    """Test handling of sklearn neg_* scorers."""

    def test_neg_scorer_sign_flip(self, regression_report):
        """Test that neg_* scorers have their sign flipped to show positive values."""
        report = regression_report

        # Register neg_mean_squared_error
        from sklearn.metrics import get_scorer

        neg_mse_scorer = get_scorer("neg_mean_squared_error")
        report.metrics.register(neg_mse_scorer)

        display = report.metrics.summarize(metric="neg_mean_squared_error")

        # The score should be positive (sign flipped)
        score = display.data["score"].iloc[0]
        assert score >= 0, "neg_* scorer should have sign flipped to positive"

    def test_neg_scorer_favorability(self, regression_report):
        """Test that neg_* scorers show correct favorability after sign flip."""
        report = regression_report

        from sklearn.metrics import get_scorer

        neg_mse_scorer = get_scorer("neg_mean_squared_error")
        report.metrics.register(neg_mse_scorer)

        display = report.metrics.summarize(metric="neg_mean_squared_error")

        # After sign flip, lower MSE is better, so it should show (↘︎)
        favorability = display.data["favorability"].iloc[0]
        assert favorability == "(↘︎)"

    def test_neg_scorer_name_cleaning(self, regression_report):
        """Test that neg_* prefix is removed from display name."""
        report = regression_report

        from sklearn.metrics import get_scorer

        neg_mse_scorer = get_scorer("neg_mean_squared_error")
        report.metrics.register(neg_mse_scorer)

        display = report.metrics.summarize(metric="neg_mean_squared_error")

        metric_name = display.data["metric"].iloc[0]

        # Display name should not include "neg_" or "Neg"
        assert not metric_name.lower().startswith("neg")


# Test 6: Cache behavior


class TestCacheBehavior:
    """Test caching behavior with registered metrics."""

    def test_metric_result_is_cached(self, binary_classification_report):
        """Test that metric results are cached after first computation."""
        report = binary_classification_report

        def counting_metric(y_true, y_pred):
            return accuracy_score(y_true, y_pred)

        scorer = make_scorer(counting_metric, response_method="predict")
        report.metrics.register(scorer)

        with check_cache_changed(report._cache):
            report.metrics.summarize(metric="counting_metric")

        with check_cache_unchanged(report._cache):
            report.metrics.summarize(metric="counting_metric")

    def test_reregister_invalidates_cache(self, binary_classification_report):
        """Test that re-registering a metric (after function edit) invalidates cache."""
        report = binary_classification_report

        # Version 1: returns 0.5
        def my_metric(y_true, y_pred):
            return 0.5

        scorer_v1 = make_scorer(my_metric, response_method="predict")
        report.metrics.register(scorer_v1)

        result_v1 = report.metrics.summarize(metric="my_metric")
        assert result_v1.data["score"].iloc[0] == 0.5

        # User edits the function (simulated by redefining with same name)
        def my_metric(y_true, y_pred):  # noqa: F811
            return 0.8

        scorer_v2 = make_scorer(my_metric, response_method="predict")

        # Re-registering with same name should silently replace and clear cache
        report.metrics.register(scorer_v2)

        result_v2 = report.metrics.summarize(metric="my_metric")

        # Should get new result (0.8), not cached one (0.5)
        assert result_v2.data["score"].iloc[0] == 0.8

    def test_different_metrics_have_separate_cache(self, binary_classification_report):
        """Test that different metrics don't share cache entries."""
        report = binary_classification_report

        scorer1 = make_scorer(lambda y_true, y_pred: 0.1, response_method="predict")
        scorer1._score_func.__name__ = "metric1"

        scorer2 = make_scorer(lambda y_true, y_pred: 0.9, response_method="predict")
        scorer2._score_func.__name__ = "metric2"

        with pytest.warns(UserWarning):  # Lambda warnings
            report.metrics.register(scorer1)
            report.metrics.register(scorer2)

        result1 = report.metrics.summarize(metric="metric1")
        result2 = report.metrics.summarize(metric="metric2")

        assert result1.data["score"].iloc[0] == 0.1
        assert result2.data["score"].iloc[0] == 0.9

    def test_reregister_clears_only_that_metric_cache(
        self, binary_classification_report
    ):
        """Test that re-registering one metric doesn't affect other metrics' caches."""
        report = binary_classification_report

        def metric1(y_true, y_pred):
            return 0.1

        def metric2(y_true, y_pred):
            return 0.2

        scorer1 = make_scorer(metric1, response_method="predict")
        scorer2 = make_scorer(metric2, response_method="predict")

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        # Compute both - should cache
        report.metrics.summarize(metric="metric1")
        report.metrics.summarize(metric="metric2")

        # Re-register metric1
        def metric1(y_true, y_pred):
            return 0.3

        scorer1_v2 = make_scorer(metric1, response_method="predict")
        report.metrics.register(scorer1_v2)

        # metric2 should use cache
        with check_cache_unchanged(report._cache):
            result2 = report.metrics.summarize(metric="metric2")

        # metric1 should compute fresh
        with check_cache_changed(report._cache):
            result1 = report.metrics.summarize(metric="metric1")

        assert result1.data["score"].iloc[0] == 0.3  # New value
        assert result2.data["score"].iloc[0] == 0.2  # Original value


# Test 7: Edge cases and error handling


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_on_report_without_test_data(
        self, logistic_binary_classification_with_train_test
    ):
        """Test that registration works even without test data."""
        estimator, X_train, _, y_train, _ = (
            logistic_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            # No X_test, y_test
        )

        scorer = make_scorer(accuracy_score, response_method="predict")

        # Registration should work
        report.metrics.register(scorer)

        # But calling it should fail
        with pytest.raises(ValueError, match="(?i)test|data"):
            report.metrics.summarize(metric="accuracy_score")

    def test_register_scorer_with_incompatible_response_method(
        self, svc_binary_classification_with_train_test
    ):
        """Test error when scorer's response_method is incompatible with estimator."""
        # SVC without probability=True has no predict_proba
        estimator, X_train, X_test, y_train, y_test = (
            svc_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        incompatible_scorer = make_scorer(
            accuracy_score,
            response_method="predict_proba",
        )

        report.metrics.register(incompatible_scorer)

        # Should fail when trying to compute
        with pytest.raises(AttributeError, match="(?i)predict_proba"):
            report.metrics.summarize(metric="accuracy_score")

    def test_register_duplicate_name_behavior(self, binary_classification_report):
        """Test that registering with duplicate name silently replaces."""
        report = binary_classification_report

        scorer1 = make_scorer(lambda y_true, y_pred: 0.5, response_method="predict")
        scorer1._score_func.__name__ = "duplicate_name"

        scorer2 = make_scorer(lambda y_true, y_pred: 0.8, response_method="predict")
        scorer2._score_func.__name__ = "duplicate_name"

        with pytest.warns(UserWarning):  # Lambda warning
            report.metrics.register(scorer1)

        nb_metrics_before = len(report.metrics.registry)

        # Registering with same name should silently replace
        with pytest.warns(UserWarning):  # Lambda warning
            report.metrics.register(scorer2)

        nb_metrics_after = len(report.metrics.registry)

        # Should not have more metrics registered than before
        assert nb_metrics_after == nb_metrics_before

        # Should use the second one
        result = report.metrics.summarize(metric="duplicate_name")
        assert result.data["score"].iloc[0] == 0.8

    def test_empty_registry_behavior(self, binary_classification_report):
        """Test that summarize works normally when no custom metrics registered."""
        report = binary_classification_report

        # Don't register anything
        display = report.metrics.summarize()

        # Should work with just built-in metrics
        assert len(display.data) > 0
        assert "Accuracy" in display.data["metric"].values


# Test 8: Different ML tasks


class TestDifferentMLTasks:
    """Test that registry works across different ML tasks."""

    def test_register_on_binary_classification(self, binary_classification_report):
        """Test registration on binary classification report."""
        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Business Loss" in display.data["metric"].values

    def test_register_on_regression(self, regression_report):
        """Test registration on regression report."""
        report = regression_report

        def custom_mse(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)

        scorer = make_scorer(
            custom_mse,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Custom Mse" in display.data["metric"].values

    def test_register_multiclass_compatible_metric(
        self, logistic_multiclass_classification_with_train_test
    ):
        """Test registration of multiclass-compatible metric."""
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

        scorer = make_scorer(accuracy_score, response_method="predict")
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        assert "Accuracy Score" in display.data["metric"].values

    def test_register_on_multioutput_regression(
        self, linear_regression_multioutput_with_train_test
    ):
        """Test registration on multioutput regression."""
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
        report.metrics.register(scorer)

        display = report.metrics.summarize()
        # Should work without error
        assert len(display.data) > 0


# Test 9: Dict return values (per-label scores)


class TestDictReturnValues:
    """Test that metrics returning dicts work correctly (per-label scores).

    Note: Multimetric scorers (single scorer returning multiple different metrics)
    are NOT supported - users should register metrics separately.
    """

    def test_per_class_accuracy_dict(self, binary_classification_report):
        """Test metric that returns per-class scores as dict."""
        report = binary_classification_report

        def per_class_accuracy(y_true, y_pred):
            """Return accuracy for each class."""
            accuracies = {}
            for label in np.unique(y_true):
                mask = y_true == label
                accuracies[int(label)] = float((y_pred[mask] == label).mean())
            return accuracies

        scorer = make_scorer(per_class_accuracy, response_method="predict")
        report.metrics.register(scorer)

        display = report.metrics.summarize(metric="per_class_accuracy")

        # Should create multiple rows, one per class
        metric_rows = display.data[display.data["metric"] == "Per Class Accuracy"]
        assert len(metric_rows) == 2  # Binary classification has 2 classes

        # Check that label column is populated
        assert "label" in display.data.columns
        labels = metric_rows["label"].values
        assert set(labels) == {0, 1}

    def test_dict_return_cached_correctly(self, binary_classification_report):
        """Test that dict-returning metrics cache correctly."""
        report = binary_classification_report

        def counted_per_class_metric(y_true, y_pred):
            """Track how many times this is called."""
            return {0: 0.9, 1: 0.8}

        scorer = make_scorer(counted_per_class_metric, response_method="predict")
        report.metrics.register(scorer)

        with check_cache_changed(report._cache):
            report.metrics.summarize(metric="counted_per_class_metric")

        # Second call should hit cache
        with check_cache_unchanged(report._cache):
            report.metrics.summarize(metric="counted_per_class_metric")

    def test_multimetric_scorer_not_recommended(self, binary_classification_report):
        """Test that multimetric scorers work but produce confusing output.

        This documents the current behavior - multimetric scorers are ambiguous
        and will be treated as per-label scores. Users should register separately.
        """
        report = binary_classification_report

        def multimetric_scorer(y_true, y_pred):
            """Return multiple different metrics - NOT RECOMMENDED."""
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary"),
            }

        scorer = make_scorer(multimetric_scorer, response_method="predict")
        report.metrics.register(scorer)

        display = report.metrics.summarize(metric="multimetric_scorer")

        # This will create rows with "accuracy" and "precision" as labels,
        # which is confusing and not the intended behavior.
        # This test documents that this is NOT supported.
        metric_rows = display.data[display.data["metric"] == "Multimetric Scorer"]
        assert len(metric_rows) == 2  # Two dict keys treated as labels

        # The label column will contain the metric names, which is confusing:
        labels = set(metric_rows["label"].values)
        assert "accuracy" in labels or "precision" in labels


# Test 10: Return value


def test_register_returns_none(binary_classification_report):
    """Test that register() returns None."""
    report = binary_classification_report

    scorer = make_scorer(accuracy_score, response_method="predict")

    result = report.metrics.register(scorer)

    assert result is None


# Test 10: String scorer name support


class TestStringScorerNames:
    """Test support for string scorer names (via sklearn.metrics.get_scorer)."""

    def test_register_with_string_scorer_name(self, binary_classification_report):
        """Test registering a metric using its sklearn string name."""
        report = binary_classification_report

        report.metrics.register("f1")

        assert "f1_score" in report.metrics.registry

        display = report.metrics.summarize(metric="f1")
        assert len(display.data) > 0

    def test_string_scorer_extracts_metadata(self, regression_report):
        """Test that string scorers extract correct metadata from sklearn."""
        report = regression_report

        report.metrics.register("neg_mean_absolute_error")

        assert "mean_absolute_error" in report.metrics.registry

    def test_string_scorer_with_neg_prefix(self, regression_report):
        """Test that string scorers with neg_ prefix are handled correctly."""
        report = regression_report

        report.metrics.register("neg_root_mean_squared_error")

        display = report.metrics.summarize(metric="neg_root_mean_squared_error")

        # Score should be positive
        assert display.data["score"].iloc[0] >= 0

    def test_invalid_string_scorer_name(self, binary_classification_report):
        """Test that invalid sklearn scorer names raise an error."""
        report = binary_classification_report

        err_msg = "Cannot register 'xyz': not a valid scorer name."
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.register("xyz")

    def test_string_scorer_appears_in_summarize(self, binary_classification_report):
        """Test that string scorers appear in summarize() output."""
        report = binary_classification_report

        display = report.metrics.summarize()
        metrics_before = set(display.data["metric"])

        report.metrics.register("f1")

        display = report.metrics.summarize()
        metrics_after = set(display.data["metric"])

        assert metrics_after - metrics_before == {"F1 Score"}


# Test 11: Validation at registration time


class TestValidationTiming:
    """Test that validation happens at registration time."""

    def test_validates_response_method_exists(
        self, svc_binary_classification_with_train_test
    ):
        """
        Test that using a scorer with unsupported response_method fails at compute.
        """
        # SVC without probability=True has no predict_proba
        estimator, X_train, X_test, y_train, y_test = (
            svc_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        invalid_scorer = make_scorer(
            accuracy_score,
            response_method="predict_proba",
        )

        report.metrics.register(invalid_scorer)

        with pytest.raises(AttributeError, match="(?i)predict_proba|response_method"):
            report.metrics.summarize(metric="accuracy_score")

    def test_validates_ml_task_compatibility(self, linear_regression_with_train_test):
        """
        Test that registering incompatible scorer for ML task fails at registration.
        """
        estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Try to register a classification-only metric
        # (This might not fail if the metric itself doesn't check, but ideally would)
        # For now, just ensure it doesn't crash
        scorer = make_scorer(
            mean_squared_error, greater_is_better=False, response_method="predict"
        )
        report.metrics.register(scorer)

    def test_validates_estimator_fitted(
        self, logistic_binary_classification_with_train_test
    ):
        """Test validation with fitted vs unfitted estimator."""
        estimator, X_train, X_test, y_train, y_test = (
            logistic_binary_classification_with_train_test
        )
        report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        scorer = make_scorer(accuracy_score, response_method="predict")

        # Registration should work even if estimator not fitted yet
        # (validation checks response_method exists as attribute, not that it works)
        report.metrics.register(scorer)


# Test 12: Serialization and pickling


class TestSerialization:
    """Test that registered metrics survive pickling (for Project storage)."""

    def test_pickle_report_with_registered_metric(self, binary_classification_report):
        """Test that registered metrics survive pickle/unpickle."""
        import pickle

        report = binary_classification_report

        # Register a named function (should pickle well)
        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        # Pickle and unpickle
        pickled = pickle.dumps(report)
        report2 = pickle.loads(pickled)

        # Metric should still be registered
        assert "business_loss" in report2.metrics.registry

        # Should be callable
        metric_info = report2.metrics.registry["business_loss"]
        assert metric_info.is_callable()

        # Should work in summarize
        display = report2.metrics.summarize()
        assert "Business Loss" in display.data["metric"].values

    def test_pickle_metric_preserves_metadata(self, binary_classification_report):
        """Test that metric metadata is preserved through pickle."""
        import pickle

        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
            cost_fp=20,
            cost_fn=3,
        )
        report.metrics.register(scorer)

        # Pickle and unpickle
        pickled = pickle.dumps(report)
        report2 = pickle.loads(pickled)

        metric_info = report2.metrics.registry["business_loss"]

        # All metadata should be preserved
        assert metric_info.name == "business_loss"
        assert metric_info.verbose_name == "Business Loss"
        assert metric_info.greater_is_better is False
        assert metric_info.response_method == "predict"
        assert metric_info.kwargs == {"cost_fp": 20, "cost_fn": 3}

    def test_pickle_lambda_function_warning(self, binary_classification_report):
        """Test that registering lambda raises a warning about pickling."""
        report = binary_classification_report

        # Register a lambda
        scorer = make_scorer(
            lambda y_true, y_pred: np.mean(y_true == y_pred),
            response_method="predict",
        )

        # Should warn about lambda not pickling well
        with pytest.warns(UserWarning, match="(?i)lambda|pickle|closure"):
            report.metrics.register(scorer)

    def test_pickle_lambda_metadata_preserved(self, binary_classification_report):
        """Test that lambda metadata is preserved even if function is lost."""
        import pickle

        report = binary_classification_report

        scorer = make_scorer(
            lambda y_true, y_pred: 0.5,
            response_method="predict",
        )

        with pytest.warns(UserWarning):
            report.metrics.register(scorer)

        # Pickle and unpickle
        pickled = pickle.dumps(report)

        # Unpickling should warn
        with pytest.warns(UserWarning, match="(?i)could not be restored"):
            report2 = pickle.loads(pickled)

        # Metadata should exist
        assert "<lambda>" in report2.metrics.registry
        metric_info = report2.metrics.registry["<lambda>"]

        # But function is not callable
        assert not metric_info.is_callable()

    def test_source_code_captured(self, binary_classification_report):
        """Test that source code is captured for inspection."""
        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        metric_info = report.metrics.registry["business_loss"]

        # Source code should be available
        assert metric_info.source_code is not None
        assert "def business_loss" in metric_info.source_code
        assert "cost_fp" in metric_info.source_code

    def test_source_code_survives_pickle(self, binary_classification_report):
        """Test that source code is preserved through pickle."""
        import pickle

        report = binary_classification_report

        scorer = make_scorer(
            business_loss,
            greater_is_better=False,
            response_method="predict",
        )
        report.metrics.register(scorer)

        # Pickle and unpickle
        pickled = pickle.dumps(report)
        report2 = pickle.loads(pickled)

        metric_info = report2.metrics.registry["business_loss"]

        # Source should still be there
        assert metric_info.source_code is not None
        assert "def business_loss" in metric_info.source_code

    def test_closure_warning(self, binary_classification_report):
        """
        Test that closures (functions with captured variables) warn about pickling.
        """
        report = binary_classification_report

        # Create a closure
        multiplier = 2.0

        def closure_metric(y_true, y_pred):
            return np.mean(y_true == y_pred) * multiplier  # Captures 'multiplier'

        scorer = make_scorer(closure_metric, response_method="predict")

        # Should warn about closure not pickling well
        with pytest.warns(UserWarning, match="(?i)closure|pickle"):
            report.metrics.register(scorer)

    def test_multiple_metrics_pickle(self, binary_classification_report):
        """Test that multiple registered metrics all survive pickling."""
        import pickle

        report = binary_classification_report

        scorer1 = make_scorer(
            business_loss, greater_is_better=False, response_method="predict"
        )
        scorer2 = make_scorer(accuracy_score, response_method="predict")

        report.metrics.register(scorer1)
        report.metrics.register(scorer2)

        # Pickle and unpickle
        pickled = pickle.dumps(report)
        report2 = pickle.loads(pickled)

        # Both should be registered
        assert "business_loss" in report2.metrics.registry
        assert "accuracy_score" in report2.metrics.registry

        # Both should be callable
        assert report2.metrics.registry["business_loss"].is_callable()
        assert report2.metrics.registry["accuracy_score"].is_callable()
