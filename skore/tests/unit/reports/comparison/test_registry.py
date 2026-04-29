"""Tests for ``ComparisonReport.metrics`` registry."""

import re

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    get_scorer,
    make_scorer,
    mean_squared_error,
    precision_score,
)

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


def business_loss_scorer(estimator, X, y, cost_fp, cost_fn):
    """Custom (estimator, X, y) scorer with required kwargs."""
    y_pred = estimator.predict(X)
    fp = ((y_pred == 1) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()
    return fp * cost_fp + fn * cost_fn


custom_scorer = make_scorer(accuracy_score, response_method="predict")


def _all_leaf_registries(report):
    """Yield every leaf ``EstimatorReport._metric_registry`` inside a comparison."""
    for sub in report.reports_.values():
        if hasattr(sub, "estimator_reports_"):
            for est in sub.estimator_reports_:
                yield est._metric_registry
        else:
            yield sub._metric_registry


@pytest.fixture(params=["estimator", "cross_validation"])
def binary_comparison_report(
    request,
    binary_classification_train_test_split,
    binary_classification_data,
):
    if request.param == "estimator":
        X_train, X_test, y_train, y_test = binary_classification_train_test_split
        return ComparisonReport(
            {
                "m1": EstimatorReport(
                    LogisticRegression(C=1.0),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                ),
                "m2": EstimatorReport(
                    LogisticRegression(C=2.0),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                ),
            }
        )
    X, y = binary_classification_data
    return ComparisonReport(
        {
            "m1": CrossValidationReport(
                LogisticRegression(C=1.0), X=X, y=y, splitter=2
            ),
            "m2": CrossValidationReport(
                LogisticRegression(C=2.0), X=X, y=y, splitter=2
            ),
        }
    )


@pytest.fixture(params=["estimator", "cross_validation"])
def regression_comparison_report(
    request,
    linear_regression_comparison_report,
    cross_validation_reports_regression,
):
    if request.param == "estimator":
        return linear_regression_comparison_report
    cv_1, cv_2 = cross_validation_reports_regression
    return ComparisonReport({"m1": cv_1, "m2": cv_2})


class TestBasicAdd:
    def test_sklearn_scorer(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy_score" in registry

    def test_callable_with_kwargs(self, binary_comparison_report):
        binary_comparison_report.metrics.add(
            business_loss_scorer, cost_fp=10, cost_fn=5
        )
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "business_loss_scorer" in registry
            assert registry["business_loss_scorer"].kwargs == {
                "cost_fp": 10,
                "cost_fn": 5,
            }

    def test_callable_missing_kwargs(self, binary_comparison_report):
        err_msg = re.escape(
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
        )
        with pytest.raises(ValueError, match=err_msg):
            binary_comparison_report.metrics.add(business_loss_scorer)

    def test_custom_name(self, binary_comparison_report):
        binary_comparison_report.metrics.add(
            business_loss_scorer, name="biz", cost_fp=1, cost_fn=2
        )
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "biz" in registry
            assert registry["biz"].verbose_name == "Biz"

    def test_multiple_metrics(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        binary_comparison_report.metrics.add(
            make_scorer(precision_score, average="macro"), name="precision_macro"
        )
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy_score" in registry
            assert "precision_macro" in registry

    def test_cannot_override_builtin(self, binary_comparison_report):
        def accuracy(y_true, y_pred):
            return 0.0

        with pytest.raises(
            ValueError,
            match="Cannot add 'accuracy': it is a built-in metric name.",
        ):
            binary_comparison_report.metrics.add(make_scorer(accuracy))

    def test_duplicate_add_raises(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot add 'accuracy_score': it already exists. "
                "Remove it first using the `remove` method."
            ),
        ):
            binary_comparison_report.metrics.add(custom_scorer)


class TestRemove:
    def test_remove_custom_metric(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy_score" in registry

        binary_comparison_report.metrics.remove("accuracy_score")

        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy_score" not in registry

    def test_remove_builtin_metric(self, binary_comparison_report):
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy" in registry

        binary_comparison_report.metrics.remove("accuracy")

        for registry in _all_leaf_registries(binary_comparison_report):
            assert "accuracy" not in registry

    def test_remove_unknown_raises(self, binary_comparison_report):
        with pytest.raises(KeyError) as exc_info:
            binary_comparison_report.metrics.remove("no_such_metric")
        assert exc_info.value.args[0] == "no_such_metric"


class TestAddPosition:
    def test_position_first(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer, position="first")
        for registry in _all_leaf_registries(binary_comparison_report):
            assert next(iter(registry.keys())) == "accuracy_score"

    def test_position_last(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer, position="last")
        for registry in _all_leaf_registries(binary_comparison_report):
            assert tuple(registry.keys())[-1] == "accuracy_score"

    def test_invalid_position(self, binary_comparison_report):
        with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
            binary_comparison_report.metrics.add(
                custom_scorer,
                position="middle",  # type: ignore[arg-type]
            )


class TestSummarizeIntegration:
    def test_summarize_includes_added_metric(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        display = binary_comparison_report.metrics.summarize()
        names = set(display.data["metric_verbose_name"])
        assert "Accuracy" in names
        assert "Accuracy Score" in names

    def test_summarize_explicit_custom_metric(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        display = binary_comparison_report.metrics.summarize(metric="accuracy_score")
        assert set(display.data["estimator_name"]) == {"m1", "m2"}
        assert set(display.data["metric_verbose_name"]) == {"Accuracy Score"}

    def test_summarize_mixed_metrics(self, binary_comparison_report):
        binary_comparison_report.metrics.add(custom_scorer)
        display = binary_comparison_report.metrics.summarize(
            metric=["accuracy", "accuracy_score"]
        )
        assert set(display.data["metric_verbose_name"]) == {
            "Accuracy",
            "Accuracy Score",
        }


class TestStringScorerNames:
    def test_add_via_string(self, binary_comparison_report):
        binary_comparison_report.metrics.add("f1")
        for registry in _all_leaf_registries(binary_comparison_report):
            assert "f1" in registry

    def test_neg_scorer(self, regression_comparison_report):
        regression_comparison_report.metrics.add(get_scorer("neg_mean_squared_error"))
        for registry in _all_leaf_registries(regression_comparison_report):
            assert "mean_squared_error" in registry
            assert registry["mean_squared_error"].greater_is_better is False

    def test_alias_without_neg_prefix(self, regression_comparison_report):
        regression_comparison_report.metrics.add("mean_squared_error")
        for registry in _all_leaf_registries(regression_comparison_report):
            assert "mean_squared_error" in registry

    def test_invalid_string_scorer_name(self, binary_comparison_report):
        with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
            binary_comparison_report.metrics.add("xyz")


class TestDifferentMLTasks:
    def test_regression(self, regression_comparison_report):
        scorer = make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
        regression_comparison_report.metrics.add(scorer)
        for registry in _all_leaf_registries(regression_comparison_report):
            assert "mean_squared_error" in registry
        display = regression_comparison_report.metrics.summarize()
        assert "Mean Squared Error" in set(display.data["metric_verbose_name"])
