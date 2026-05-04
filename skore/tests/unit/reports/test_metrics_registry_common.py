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
    """Yield every ``EstimatorReport._metric_registry`` reachable from ``report``."""
    if hasattr(report, "estimator_reports_"):
        for sub in report.estimator_reports_:
            yield sub._metric_registry
    elif hasattr(report, "reports_"):
        for sub in report.reports_.values():
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
def regression_report(request):
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
def multioutput_regression_report(request):
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


class TestBasicAdd:
    """Add propagates to every leaf ``EstimatorReport._metric_registry``."""

    def test_sklearn_scorer(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        for registry in leaf_registries(binary_report):
            assert "accuracy_score" in registry

    def test_callable_with_kwargs(self, binary_report):
        binary_report.metrics.add(business_loss_scorer, cost_fp=20, cost_fn=3)
        for registry in leaf_registries(binary_report):
            assert "business_loss_scorer" in registry
            assert registry["business_loss_scorer"].kwargs == {
                "cost_fp": 20,
                "cost_fn": 3,
            }
        display = binary_report.metrics.summarize(metric="business_loss_scorer")
        assert display.data["score"].notna().all()

    def test_callable_with_name(self, binary_report):
        binary_report.metrics.add(
            business_loss_scorer, name="custom_metric", cost_fp=10, cost_fn=5
        )
        for registry in leaf_registries(binary_report):
            assert "custom_metric" in registry
            assert registry["custom_metric"].verbose_name == "Custom Metric"

    def test_metric_instance(self, binary_report):
        metric = Metric.new(get_scorer("accuracy"), name="custom_acc")
        binary_report.metrics.add(metric)
        for registry in leaf_registries(binary_report):
            assert "custom_acc" in registry
        display = binary_report.metrics.summarize(metric="custom_acc")
        assert display.data["score"].notna().all()

    def test_multiple_metrics(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        binary_report.metrics.add(
            make_scorer(precision_score, average="macro"), name="precision_macro"
        )
        for registry in leaf_registries(binary_report):
            assert "accuracy_score" in registry
            assert "precision_macro" in registry

    def test_cannot_override_builtin(self, binary_report):
        def accuracy(y_true, y_pred):
            return 0.0

        with pytest.raises(
            ValueError,
            match="Cannot add 'accuracy': it is a built-in metric name.",
        ):
            binary_report.metrics.add(make_scorer(accuracy))

    def test_duplicate_add_raises(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot add 'accuracy_score': it already exists. "
                "Remove it first using the `remove` method."
            ),
        ):
            binary_report.metrics.add(custom_scorer)


class TestRemove:
    """Remove drops the metric from every leaf registry."""

    def test_remove_custom_metric(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        for registry in leaf_registries(binary_report):
            assert "accuracy_score" in registry

        binary_report.metrics.remove("accuracy_score")

        for registry in leaf_registries(binary_report):
            assert "accuracy_score" not in registry

    def test_remove_builtin_metric(self, binary_report):
        for registry in leaf_registries(binary_report):
            assert "accuracy" in registry

        binary_report.metrics.remove("accuracy")

        for registry in leaf_registries(binary_report):
            assert "accuracy" not in registry
        frame = binary_report.metrics.summarize().frame()
        assert "Accuracy" not in frame.index

    def test_remove_unknown_raises(self, binary_report):
        with pytest.raises(KeyError) as exc_info:
            binary_report.metrics.remove("no_such_metric")
        assert exc_info.value.args[0] == "no_such_metric"


class TestAddPosition:
    """Position kwarg controls the placement in every leaf registry."""

    def test_position_first(self, binary_report):
        binary_report.metrics.add(custom_scorer, position="first")
        for registry in leaf_registries(binary_report):
            assert next(iter(registry.keys())) == "accuracy_score"

    def test_position_last(self, binary_report):
        binary_report.metrics.add(custom_scorer, position="last")
        for registry in leaf_registries(binary_report):
            assert tuple(registry.keys())[-1] == "accuracy_score"

    def test_invalid_position(self, binary_report):
        with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
            binary_report.metrics.add(
                custom_scorer,
                position="middle",  # type: ignore[arg-type]
            )

    def test_mixed_first_and_last(self, binary_report):
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


class TestSummarizeIntegration:
    """Added metrics show up in ``summarize()`` output."""

    def test_summarize_includes_added_metric(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        display = binary_report.metrics.summarize()
        names = set(display.data["metric_verbose_name"])
        assert "Accuracy" in names
        assert "Accuracy Score" in names

    def test_summarize_with_mixed_metrics(self, binary_report):
        binary_report.metrics.add(custom_scorer)
        display = binary_report.metrics.summarize(metric=["accuracy", "accuracy_score"])
        assert set(display.data["metric_verbose_name"]) == {
            "Accuracy",
            "Accuracy Score",
        }


class TestStringScorerNames:
    """String scorer names (resolved via sklearn) work for every report kind."""

    def test_add_via_string(self, binary_report):
        binary_report.metrics.add("f1")
        for registry in leaf_registries(binary_report):
            assert "f1" in registry

    def test_string_scorer_appears_in_summarize(self, binary_report):
        display = binary_report.metrics.summarize()
        metrics_before = set(display.data["metric_verbose_name"])

        binary_report.metrics.add("balanced_accuracy")

        display = binary_report.metrics.summarize()
        metrics_after = set(display.data["metric_verbose_name"])

        assert metrics_after - metrics_before == {"Balanced Accuracy"}

    def test_neg_scorer(self, regression_report):
        """``neg_*`` scorers strip the prefix and flip ``greater_is_better``."""
        regression_report.metrics.add(get_scorer("neg_mean_squared_error"))
        for registry in leaf_registries(regression_report):
            assert "mean_squared_error" in registry
            metric = registry["mean_squared_error"]
            assert metric.greater_is_better is False
            assert not metric.verbose_name.lower().startswith("neg")

    def test_alias_without_neg_prefix(self, regression_report):
        regression_report.metrics.add("mean_squared_error")
        for registry in leaf_registries(regression_report):
            assert "mean_squared_error" in registry

    def test_invalid_string_scorer_name(self, binary_report):
        with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
            binary_report.metrics.add("xyz")


class TestDifferentMLTasks:
    """The registry is consistent across binary / multiclass / regression tasks."""

    def test_multiclass(self, multiclass_report):
        multiclass_report.metrics.add(custom_scorer)
        for registry in leaf_registries(multiclass_report):
            assert "accuracy_score" in registry
        display = multiclass_report.metrics.summarize()
        assert "Accuracy Score" in set(display.data["metric_verbose_name"])

    def test_regression(self, regression_report):
        scorer = make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
        regression_report.metrics.add(scorer)
        for registry in leaf_registries(regression_report):
            assert "mean_squared_error" in registry
        display = regression_report.metrics.summarize()
        assert "Mean Squared Error" in set(display.data["metric_verbose_name"])

    def test_multioutput_regression(self, multioutput_regression_report):
        scorer = make_scorer(
            mean_squared_error,
            greater_is_better=False,
            response_method="predict",
        )
        multioutput_regression_report.metrics.add(scorer)
        for registry in leaf_registries(multioutput_regression_report):
            assert "mean_squared_error" in registry


class TestDictReturnValues:
    """Metrics returning dicts expand into per-label rows.

    Multimetric scorers (single scorer returning multiple different metrics) are
    NOT supported - users should add metrics separately. We test that the current
    behavior exposes them as per-label rows so the contract is documented.
    """

    def test_per_class_accuracy_dict(self, binary_report):
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
        metric_rows = display.data[
            display.data["metric_verbose_name"] == "Per Class Accuracy"
        ]
        assert set(metric_rows["label"].values) == {0, 1}

    def test_multimetric_scorer_not_recommended(self, binary_report):
        def multimetric_scorer(y_true, y_pred):
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary"),
            }

        binary_report.metrics.add(
            make_scorer(multimetric_scorer, response_method="predict")
        )

        display = binary_report.metrics.summarize(metric="multimetric_scorer")
        metric_rows = display.data[
            display.data["metric_verbose_name"] == "Multimetric Scorer"
        ]
        assert set(metric_rows["label"]) == {"accuracy", "precision"}
