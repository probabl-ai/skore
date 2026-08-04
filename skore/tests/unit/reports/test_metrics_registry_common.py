"""Tests for the metrics registry common to every report kind."""

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

from skore._sklearn.metrics import Metric


def business_loss_scorer(estimator, X, y, cost_fp, cost_fn):
    """Custom (estimator, X, y) scorer with required kwargs."""
    y_pred = estimator.predict(X)
    fp = ((y_pred == 1) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()
    return fp * cost_fp + fn * cost_fn


custom_scorer = make_scorer(accuracy_score, response_method="predict")


def leaf_registries(report):
    """Yield every ``EstimatorReport._metric_registry`` reachable from ``report``.

    ``CrossValidationReport`` stores children in ``reports_`` (list) or, on older
    layouts, ``estimator_reports_``. ``ComparisonReport`` stores children in
    ``reports_`` (dict).
    """
    if hasattr(report, "estimator_reports_"):
        for sub in report.estimator_reports_:
            yield sub._metric_registry
    elif hasattr(report, "reports_"):
        children = report.reports_
        if isinstance(children, dict):
            children = children.values()
        for sub in children:
            yield from leaf_registries(sub)
    else:
        yield report._metric_registry


_REPORT_KINDS = [
    "estimator",
    "cross_validation",
    "comparison_estimator",
    "comparison_cv",
]


def _resolve_report(request, fixture_map):
    obj = request.getfixturevalue(fixture_map[request.param])
    return obj[0] if isinstance(obj, tuple) else obj


@pytest.fixture(params=_REPORT_KINDS)
def binary_report(request):
    return _resolve_report(
        request,
        {
            "estimator": "estimator_reports_binary_classification",
            "cross_validation": "cross_validation_report_binary_classification",
            "comparison_estimator": (
                "comparison_estimator_reports_binary_classification"
            ),
            "comparison_cv": (
                "comparison_cross_validation_reports_binary_classification"
            ),
        },
    )


@pytest.fixture(params=_REPORT_KINDS)
def regression_kind_report(request):
    return _resolve_report(
        request,
        {
            "estimator": "estimator_reports_regression",
            "cross_validation": "cross_validation_reports_regression",
            "comparison_estimator": "comparison_estimator_reports_regression",
            "comparison_cv": "comparison_cross_validation_reports_regression",
        },
    )


@pytest.fixture(params=_REPORT_KINDS)
def multiclass_report(request):
    return _resolve_report(
        request,
        {
            "estimator": "estimator_reports_multiclass_classification",
            "cross_validation": "cross_validation_report_multiclass_classification",
            "comparison_estimator": (
                "comparison_estimator_reports_multiclass_classification"
            ),
            "comparison_cv": (
                "comparison_cross_validation_reports_multiclass_classification"
            ),
        },
    )


@pytest.fixture(params=_REPORT_KINDS)
def multioutput_kind_report(request):
    return _resolve_report(
        request,
        {
            "estimator": "estimator_reports_multioutput_regression",
            "cross_validation": "cross_validation_reports_multioutput_regression",
            "comparison_estimator": (
                "comparison_estimator_reports_multioutput_regression"
            ),
            "comparison_cv": (
                "comparison_cross_validation_reports_multioutput_regression"
            ),
        },
    )


def test_add_sklearn_scorer(binary_report):
    binary_report.metrics.add(custom_scorer)
    for registry in leaf_registries(binary_report):
        assert "accuracy_score" in registry


def test_add_callable_with_kwargs(binary_report):
    binary_report.metrics.add(business_loss_scorer, cost_fp=20, cost_fn=3)
    for registry in leaf_registries(binary_report):
        assert "business_loss_scorer" in registry
        assert registry["business_loss_scorer"].kwargs == {
            "cost_fp": 20,
            "cost_fn": 3,
        }
    display = binary_report.metrics.summarize(metric="business_loss_scorer")
    assert display.summary["score"].notna().all()


def test_add_callable_with_name(binary_report):
    binary_report.metrics.add(
        business_loss_scorer, name="custom_metric", cost_fp=10, cost_fn=5
    )
    for registry in leaf_registries(binary_report):
        assert "custom_metric" in registry
        assert registry["custom_metric"].verbose_name == "Custom Metric"


def test_add_callable_with_verbose_name(binary_report):
    binary_report.metrics.add(
        business_loss_scorer, verbose_name="hello", cost_fp=10, cost_fn=5
    )
    for registry in leaf_registries(binary_report):
        assert "business_loss_scorer" in registry
        assert registry["business_loss_scorer"].verbose_name == "hello"


def test_add_metric_instance(binary_report):
    metric = Metric.new(get_scorer("accuracy"))
    binary_report.metrics.add(metric, name="custom_acc")
    for registry in leaf_registries(binary_report):
        assert "custom_acc" in registry
    display = binary_report.metrics.summarize(metric="custom_acc")
    assert display.summary["score"].notna().all()


def test_add_metric_instance_with_verbose_name(binary_report):
    metric = Metric.new(get_scorer("accuracy"))
    binary_report.metrics.add(metric, verbose_name="custom_acc")
    for registry in leaf_registries(binary_report):
        assert "accuracy_score" in registry
    display = binary_report.metrics.summarize(metric="accuracy_score")
    assert set(display.summary["verbose_name"]) == {"custom_acc"}


def test_add_multiple_metrics(binary_report):
    binary_report.metrics.add(custom_scorer)
    binary_report.metrics.add(
        make_scorer(precision_score, average="macro"), name="precision_macro_custom"
    )
    for registry in leaf_registries(binary_report):
        assert "accuracy_score" in registry
        assert "precision_macro_custom" in registry


def test_add_cannot_override_builtin(binary_report):
    def accuracy(y_true, y_pred):
        return 0.0

    with pytest.raises(
        ValueError,
        match="Cannot add 'accuracy': it is a built-in metric name.",
    ):
        binary_report.metrics.add(make_scorer(accuracy))


def test_add_duplicate_raises(binary_report):
    binary_report.metrics.add(custom_scorer)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot add 'accuracy_score': it already exists. "
            "Remove it first using the `remove` method."
        ),
    ):
        binary_report.metrics.add(custom_scorer)


def test_remove_custom_metric(binary_report):
    binary_report.metrics.add(custom_scorer)
    for registry in leaf_registries(binary_report):
        assert "accuracy_score" in registry

    binary_report.metrics.remove("accuracy_score")

    for registry in leaf_registries(binary_report):
        assert "accuracy_score" not in registry


def test_remove_builtin_metric(binary_report):
    for registry in leaf_registries(binary_report):
        assert "accuracy" in registry

    binary_report.metrics.remove("accuracy")

    for registry in leaf_registries(binary_report):
        assert "accuracy" not in registry
    display = binary_report.metrics.summarize()
    assert "Accuracy" not in set(display.summary["verbose_name"])


def test_remove_unknown_raises(binary_report):
    with pytest.raises(KeyError) as exc_info:
        binary_report.metrics.remove("no_such_metric")
    assert exc_info.value.args[0] == "no_such_metric"


def test_position_first(binary_report):
    binary_report.metrics.add(custom_scorer, position="first")
    for registry in leaf_registries(binary_report):
        assert next(iter(registry.keys())) == "accuracy_score"


def test_position_last(binary_report):
    binary_report.metrics.add(custom_scorer, position="last")
    for registry in leaf_registries(binary_report):
        assert tuple(registry.keys())[-1] == "accuracy_score"


def test_position_invalid(binary_report):
    with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
        binary_report.metrics.add(
            custom_scorer,
            position="middle",  # type: ignore[arg-type]
        )


def test_position_mixed_first_and_last(binary_report):
    def m_first(y_true, y_pred):
        return 0.0

    def m_last(y_true, y_pred):
        return 1.0

    binary_report.metrics.add(
        make_scorer(m_first, response_method="predict"), name="m_first"
    )
    binary_report.metrics.add(
        make_scorer(m_last, response_method="predict"),
        name="m_last",
        position="last",
    )

    for registry in leaf_registries(binary_report):
        keys = tuple(registry.keys())
        assert keys[0] == "m_first"
        assert keys[-1] == "m_last"


def test_position_default_first_lifo(binary_report):
    def metric_a(y_true, y_pred):
        return 0.0

    def metric_b(y_true, y_pred):
        return 1.0

    binary_report.metrics.add(
        make_scorer(metric_a, response_method="predict"), name="metric_a"
    )
    binary_report.metrics.add(
        make_scorer(metric_b, response_method="predict"), name="metric_b"
    )

    for registry in leaf_registries(binary_report):
        keys = list(registry.keys())
        assert keys[0] == "metric_b"
        assert keys[1] == "metric_a"


def test_summarize_includes_added_metric(binary_report):
    binary_report.metrics.add(custom_scorer)
    display = binary_report.metrics.summarize()
    names = set(display.summary["verbose_name"])
    assert "Accuracy" in names
    assert "Accuracy Score" in names


def test_summarize_with_mixed_metrics(binary_report):
    binary_report.metrics.add(custom_scorer)
    display = binary_report.metrics.summarize(metric=["accuracy", "accuracy_score"])
    assert set(display.summary["verbose_name"]) == {
        "Accuracy",
        "Accuracy Score",
    }


def test_string_scorer_add(binary_report):
    binary_report.metrics.add("f1")
    for registry in leaf_registries(binary_report):
        assert "f1" in registry


def test_string_scorer_appears_in_summarize(binary_report):
    display = binary_report.metrics.summarize()
    metrics_before = set(display.summary["verbose_name"])

    binary_report.metrics.add("balanced_accuracy")

    display = binary_report.metrics.summarize()
    metrics_after = set(display.summary["verbose_name"])

    assert metrics_after - metrics_before == {"Balanced Accuracy"}


def test_neg_scorer_from_get_scorer(regression_kind_report):
    """``get_scorer("neg_*")`` strips the prefix via the underlying function name."""
    regression_kind_report.metrics.add(get_scorer("neg_mean_squared_error"))
    for registry in leaf_registries(regression_kind_report):
        assert "mean_squared_error" in registry
        metric = registry["mean_squared_error"]
        assert metric.greater_is_better is False
        assert not metric.verbose_name.lower().startswith("neg")


def test_string_scorer_alias_without_neg_prefix(regression_kind_report):
    regression_kind_report.metrics.add("mean_squared_error")
    for registry in leaf_registries(regression_kind_report):
        assert "mean_squared_error" in registry


def test_string_scorer_with_neg_prefix(regression_kind_report):
    """String names with ``neg_`` keep the user-provided name."""
    regression_kind_report.metrics.add("mean_squared_error")
    assert all(
        "mean_squared_error" in registry
        for registry in leaf_registries(regression_kind_report)
    )

    regression_kind_report.metrics.add("neg_mean_squared_error")
    for registry in leaf_registries(regression_kind_report):
        assert "neg_mean_squared_error" in registry
        assert "mean_squared_error" in registry


def test_summarize_with_neg_prefix(regression_kind_report):
    regression_kind_report.metrics.add("neg_mean_absolute_percentage_error")
    for registry in leaf_registries(regression_kind_report):
        assert "neg_mean_absolute_percentage_error" in registry
        assert "mean_absolute_percentage_error" not in registry

    regression_kind_report.metrics.summarize(
        metric="neg_mean_absolute_percentage_error"
    )

    with pytest.raises(KeyError, match="mean_absolute_percentage_error"):
        regression_kind_report.metrics.summarize(
            metric="mean_absolute_percentage_error"
        )


def test_invalid_string_scorer_name(binary_report):
    with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
        binary_report.metrics.add("xyz")


def test_multiclass_add(multiclass_report):
    multiclass_report.metrics.add(custom_scorer)
    for registry in leaf_registries(multiclass_report):
        assert "accuracy_score" in registry
    display = multiclass_report.metrics.summarize()
    assert "Accuracy Score" in set(display.summary["verbose_name"])


def test_regression_add(regression_kind_report):
    scorer = make_scorer(
        mean_squared_error,
        greater_is_better=False,
        response_method="predict",
    )
    regression_kind_report.metrics.add(scorer)
    for registry in leaf_registries(regression_kind_report):
        assert "mean_squared_error" in registry
    display = regression_kind_report.metrics.summarize()
    assert "Mean Squared Error" in set(display.summary["verbose_name"])


def test_multioutput_regression_add(multioutput_kind_report):
    scorer = make_scorer(
        mean_squared_error,
        greater_is_better=False,
        response_method="predict",
    )
    multioutput_kind_report.metrics.add(scorer)
    for registry in leaf_registries(multioutput_kind_report):
        assert "mean_squared_error" in registry


def test_dict_return_per_class_accuracy(binary_report):
    def per_class_accuracy(y_true, y_pred) -> dict[int, float]:
        accuracies: dict[int, float] = {}
        for label in np.unique(y_true):
            mask = y_true == label
            accuracies[int(label)] = float((y_pred[mask] == label).mean())
        return accuracies

    def scorer(est, X, y_true):
        y_pred = est.predict(X)
        return per_class_accuracy(y_true, y_pred)

    binary_report.metrics.add(scorer, name="per_class_accuracy")

    display = binary_report.metrics.summarize(metric="per_class_accuracy")
    assert set(display.summary["verbose_name"]) == {0, 1}


def test_multimetric_scorer_expands_submetrics(binary_report):
    def multimetric_scorer(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
        }

    binary_report.metrics.add(
        make_scorer(multimetric_scorer, response_method="predict")
    )

    display = binary_report.metrics.summarize(metric="multimetric_scorer")
    assert set(display.summary["verbose_name"]) == {"accuracy", "precision"}
