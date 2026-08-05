"""Tests for the user-facing ``report.metrics`` registry API.

The first few tests are parametrized over every report kind to check that the
accessor propagates to each leaf registry. The rest of the API is report-agnostic
and is therefore exercised once, on an ``EstimatorReport``.
"""

import math
import re

import pytest
from sklearn.metrics import (
    accuracy_score,
    get_scorer,
    make_scorer,
    mean_squared_error,
    precision_score,
)

from skore._sklearn.metrics import Metric
from tests.unit.reports._registry_helpers import business_loss_scorer

accuracy_scorer = make_scorer(accuracy_score, response_method="predict")


def leaf_registries(report):
    """Yield every ``EstimatorReport._metric_registry`` reachable from ``report``.

    ``CrossValidationReport`` stores its children in ``reports_`` as a list and
    ``ComparisonReport`` as a dict; anything else is an ``EstimatorReport`` leaf.
    """
    children = getattr(report, "reports_", None)
    if children is None:
        yield report._metric_registry
        return
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        yield from leaf_registries(child)


@pytest.fixture(
    params=["estimator", "cross_validation", "comparison_estimator", "comparison_cv"]
)
def binary_report(request):
    """Binary classification report, one per report kind.

    Only used by the tests below that check that the accessor propagates to
    every leaf registry; everything else runs once on an ``EstimatorReport``.
    """
    fixture_name = {
        "estimator": "estimator_reports_binary_classification",
        "cross_validation": "cross_validation_report_binary_classification",
        "comparison_estimator": "comparison_estimator_reports_binary_classification",
        "comparison_cv": "comparison_cross_validation_reports_binary_classification",
    }[request.param]
    report = request.getfixturevalue(fixture_name)
    return report[0] if isinstance(report, tuple) else report


def test_add_propagates_to_leaf_registries(binary_report):
    binary_report.metrics.add(accuracy_scorer)
    for registry in leaf_registries(binary_report):
        assert "accuracy_score" in registry


def test_remove_propagates_to_leaf_registries(binary_report):
    for registry in leaf_registries(binary_report):
        assert "accuracy" in registry

    binary_report.metrics.remove("accuracy")

    for registry in leaf_registries(binary_report):
        assert "accuracy" not in registry
    display = binary_report.metrics.summarize()
    assert "Accuracy" not in set(display.summary["verbose_name"])


def test_position_propagates_to_leaf_registries(binary_report):
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


def test_summarize_includes_added_metric(binary_report):
    binary_report.metrics.add(accuracy_scorer)
    display = binary_report.metrics.summarize()
    names = set(display.summary["verbose_name"])
    assert "Accuracy" in names
    assert "Accuracy Score" in names


def test_add_callable_with_kwargs(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(business_loss_scorer, cost_fp=20, cost_fn=3)

    registry = report._metric_registry
    assert registry["business_loss_scorer"].kwargs == {"cost_fp": 20, "cost_fn": 3}
    display = report.metrics.summarize(metric="business_loss_scorer")
    assert display.summary["score"].notna().all()


def test_add_callable_with_name(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(
        business_loss_scorer, name="custom_metric", cost_fp=10, cost_fn=5
    )

    assert report._metric_registry["custom_metric"].verbose_name == "Custom Metric"


def test_add_callable_with_verbose_name(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(
        business_loss_scorer, verbose_name="hello", cost_fp=10, cost_fn=5
    )

    assert report._metric_registry["business_loss_scorer"].verbose_name == "hello"


def test_add_metric_instance(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(Metric.new(get_scorer("accuracy")), name="custom_acc")

    assert "custom_acc" in report._metric_registry
    display = report.metrics.summarize(metric="custom_acc")
    assert display.summary["score"].notna().all()


def test_add_metric_instance_with_verbose_name(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(Metric.new(get_scorer("accuracy")), verbose_name="custom_acc")

    assert "accuracy_score" in report._metric_registry
    display = report.metrics.summarize(metric="accuracy_score")
    assert set(display.summary["verbose_name"]) == {"custom_acc"}


def test_add_multiple_metrics(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(accuracy_scorer)
    report.metrics.add(
        make_scorer(precision_score, average="macro"), name="precision_macro_custom"
    )

    assert "accuracy_score" in report._metric_registry
    assert "precision_macro_custom" in report._metric_registry


def test_readd_default_metric(binary_classification_report):
    """A default metric can be removed and added back under the same name."""
    report = binary_classification_report

    def accuracy(y_true, y_pred):
        return 1.0

    report.metrics.remove("accuracy")
    report.metrics.add(make_scorer(accuracy))

    assert report.metrics.get("accuracy") == 1.0


def test_add_duplicate_raises(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(accuracy_scorer)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot add 'accuracy_score': it already exists. "
            "Remove it first using the `remove` method."
        ),
    ):
        report.metrics.add(accuracy_scorer)


def test_remove_custom_metric(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(accuracy_scorer)
    assert "accuracy_score" in report._metric_registry

    report.metrics.remove("accuracy_score")

    assert "accuracy_score" not in report._metric_registry


def test_remove_unknown_raises(binary_classification_report):
    with pytest.raises(KeyError) as exc_info:
        binary_classification_report.metrics.remove("no_such_metric")
    assert exc_info.value.args[0] == "no_such_metric"


def test_position_invalid(binary_classification_report):
    with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
        binary_classification_report.metrics.add(
            accuracy_scorer,
            position="middle",  # type: ignore[arg-type]
        )


def test_position_default_first_lifo(binary_classification_report):
    report = binary_classification_report

    def metric_a(y_true, y_pred):
        return 0.0

    def metric_b(y_true, y_pred):
        return 1.0

    report.metrics.add(make_scorer(metric_a, response_method="predict"))
    report.metrics.add(make_scorer(metric_b, response_method="predict"))

    keys = list(report._metric_registry.keys())
    assert keys[0] == "metric_b"
    assert keys[1] == "metric_a"


def test_summarize_with_mixed_metrics(binary_classification_report):
    report = binary_classification_report
    report.metrics.add(accuracy_scorer)
    display = report.metrics.summarize(metric=["accuracy", "accuracy_score"])
    assert set(display.summary["verbose_name"]) == {"Accuracy", "Accuracy Score"}


def test_string_scorer_add(binary_classification_report):
    binary_classification_report.metrics.add("f1")
    assert "f1" in binary_classification_report._metric_registry


def test_string_scorer_appears_in_summarize(binary_classification_report):
    report = binary_classification_report
    metrics_before = set(report.metrics.summarize().summary["verbose_name"])

    report.metrics.add("balanced_accuracy")

    metrics_after = set(report.metrics.summarize().summary["verbose_name"])
    assert metrics_after - metrics_before == {"Balanced Accuracy"}


def test_invalid_string_scorer_name(binary_classification_report):
    with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
        binary_classification_report.metrics.add("xyz")


def test_neg_scorer_from_get_scorer(regression_report):
    """``get_scorer("neg_*")`` strips the prefix via the underlying function name."""
    regression_report.metrics.add(get_scorer("neg_mean_squared_error"))

    metric = regression_report._metric_registry["mean_squared_error"]
    assert metric.greater_is_better is False
    assert not metric.verbose_name.lower().startswith("neg")


def test_string_scorer_with_neg_prefix(regression_report):
    """A ``neg_``-prefixed name keeps its prefix and does not clash with the alias."""
    regression_report.metrics.add("mean_squared_error")
    assert "mean_squared_error" in regression_report._metric_registry

    regression_report.metrics.add("neg_mean_squared_error")
    assert "neg_mean_squared_error" in regression_report._metric_registry
    assert "mean_squared_error" in regression_report._metric_registry


def test_summarize_with_neg_prefix(regression_report):
    """``summarize`` looks metrics up under their registered name only."""
    regression_report.metrics.add("neg_mean_absolute_percentage_error")
    assert "mean_absolute_percentage_error" not in regression_report._metric_registry

    regression_report.metrics.summarize(metric="neg_mean_absolute_percentage_error")

    with pytest.raises(KeyError, match="mean_absolute_percentage_error"):
        regression_report.metrics.summarize(metric="mean_absolute_percentage_error")


def test_get_with_neg_prefix(regression_report):
    """``get`` looks metrics up under their registered name only."""
    regression_report.metrics.add("neg_mean_absolute_percentage_error")

    value = regression_report.metrics.get("neg_mean_absolute_percentage_error")
    assert math.isfinite(value)

    with pytest.raises(KeyError, match="mean_absolute_percentage_error"):
        regression_report.metrics.get("mean_absolute_percentage_error")


def test_unknown_metric_raises_key_error(regression_report):
    with pytest.raises(KeyError, match="nonexistent_metric"):
        regression_report.metrics.summarize(metric="nonexistent_metric")

    with pytest.raises(KeyError, match="nonexistent_metric"):
        regression_report.metrics.get("nonexistent_metric")


def test_multiclass_add(multiclass_classification_report):
    report = multiclass_classification_report
    report.metrics.add(accuracy_scorer)

    assert "accuracy_score" in report._metric_registry
    display = report.metrics.summarize()
    assert "Accuracy Score" in set(display.summary["verbose_name"])


def test_regression_add(regression_report):
    scorer = make_scorer(
        mean_squared_error, greater_is_better=False, response_method="predict"
    )
    regression_report.metrics.add(scorer)

    assert "mean_squared_error" in regression_report._metric_registry
    display = regression_report.metrics.summarize()
    assert "Mean Squared Error" in set(display.summary["verbose_name"])


def test_multioutput_regression_add(multioutput_regression_report):
    scorer = make_scorer(
        mean_squared_error, greater_is_better=False, response_method="predict"
    )
    multioutput_regression_report.metrics.add(scorer)

    assert "mean_squared_error" in multioutput_regression_report._metric_registry
    display = multioutput_regression_report.metrics.summarize()
    assert "Mean Squared Error" in set(display.summary["verbose_name"])
