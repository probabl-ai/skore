"""Unit tests for Metric, MetricRegistry, and related helpers."""

from __future__ import annotations

import functools
import pickle
import re

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
)

from skore import EstimatorReport
from skore._sklearn.metrics import (
    _METRIC_ALIASES,
    BUILTIN_METRICS,
    RESERVED_METRIC_NAMES,
    FunctionKind,
    Metric,
    MetricRegistry,
    MissingKwargsError,
    Score,
)
from skore._utils._cache_key import make_cache_key
from tests.unit.reports._registry_helpers import (
    business_loss_metric,
    business_loss_scorer,
)


def test_missing_kwargs_error_attributes_and_message():
    err = MissingKwargsError(business_loss_scorer, ("cost_fp", "cost_fn"))
    assert err.metric == "business_loss_scorer"
    assert err.missing_kwargs == ("cost_fp", "cost_fn")
    assert err.msg == (
        "Callable 'business_loss_scorer' has required parameter(s) "
        "('cost_fp', 'cost_fn') not covered by the provided kwargs."
    )


def test_missing_kwargs_error_partial_callable_name():
    partial = functools.partial(business_loss_scorer, cost_fp=1)
    err = MissingKwargsError(partial, ("cost_fn",))
    assert err.metric == "business_loss_scorer"


def test_metric_init_full_args():
    m = Metric(
        name="custom",
        verbose_name="My Custom",
        greater_is_better=True,
        response_method="predict",
        function=accuracy_score,
        function_kind=FunctionKind.METRIC,
        kwargs={"average": "binary"},
    )
    assert m.name == "custom"
    assert m.verbose_name == "My Custom"
    assert m.greater_is_better is True
    assert m.response_method == "predict"
    assert m.function is accuracy_score
    assert m.function_kind is FunctionKind.METRIC
    assert m.kwargs == {"average": "binary"}


def test_metric_init_default_verbose_name():
    m = Metric(name="my_metric")
    assert m.verbose_name == "My Metric"


def test_metric_init_kwargs_default():
    m = Metric(name="x")
    assert m.kwargs == {}


def test_metric_init_subclass_path_kwargs_propagated():
    m = Metric(kwargs={"average": "macro"})
    assert m.kwargs == {"average": "macro"}


def test_metric_greater_is_better_none():
    m = Metric(name="test", greater_is_better=None)
    assert m.greater_is_better is None

    m = Metric(name="test", greater_is_better=True)
    assert m.greater_is_better is True


def test_metric_getstate_drops_lambda():
    scorer = make_scorer(lambda y_true, y_pred: 0.0)
    m = Metric(
        name="drop_me",
        function=scorer._score_func,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    state = m.__getstate__()
    assert state["function"] is None


def test_metric_getstate_keeps_picklable():
    m = Metric(
        name="keep",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    state = m.__getstate__()
    assert state["function"] is accuracy_score


def test_metric_getstate_round_trip():
    m = Metric(
        name="rt",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.name == "rt"
    assert m2.function is accuracy_score


def test_metric_repr():
    m = Metric(name="accuracy", function=None, greater_is_better=True)
    assert repr(m) == (
        "Metric(name='accuracy', verbose_name='Accuracy', function=None, "
        "greater_is_better=True, response_method=None, kwargs={})"
    )


def test_metric_repr_kwargs():
    m = Metric(
        name="accuracy", function=None, greater_is_better=True, kwargs={"hello": 1}
    )
    assert repr(m) == (
        "Metric(name='accuracy', verbose_name='Accuracy', function=None, "
        "greater_is_better=True, response_method=None, kwargs={'hello': 1})"
    )


def test_metric_available_default(binary_classification_report):
    m = Metric(name="test")
    assert m.available(binary_classification_report) is True


def test_metric_rows_no_function(binary_classification_report):
    m = Metric(name="abstract_metric", function=None)
    err_msg = "Metric 'abstract_metric' has no scoring function."
    with pytest.raises(ValueError, match=err_msg):
        m.rows(report=binary_classification_report, data_source="test")


def test_metric_rows_function_kind_metric(binary_classification_report):
    m = Metric(
        name="acc",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    rows = m.rows(report=binary_classification_report, data_source="test")
    assert len(rows) == 1
    assert 0.0 <= rows[0]["score"] <= 1.0


def test_metric_rows_function_kind_scorer(binary_classification_report):
    def my_scorer(estimator, X, y_true):
        return float((estimator.predict(X) == y_true).mean())

    m = Metric(
        name="scorer_acc",
        function=my_scorer,
        greater_is_better=True,
        function_kind=FunctionKind.SCORER,
    )
    rows = m.rows(report=binary_classification_report, data_source="test")
    assert len(rows) == 1
    assert 0.0 <= rows[0]["score"] <= 1.0


def test_metric_rows_cache_hit(binary_classification_report):
    m = Metric(
        name="acc",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    sentinel = object()
    cache_key = make_cache_key("metrics", "test", "acc", {})
    binary_classification_report._cache[cache_key] = sentinel
    rows = m.rows(report=binary_classification_report, data_source="test")
    assert rows[0]["score"] is sentinel


def test_metric_rows_cache_populated(binary_classification_report):
    m = Metric(
        name="acc",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    rows = m.rows(report=binary_classification_report, data_source="test")
    cached = binary_classification_report._cache[
        make_cache_key("metrics", "test", "acc", {})
    ]
    assert cached == rows[0]["score"]


def test_metric_rows_pos_label_injection(binary_classification_report):
    captured = {}

    def metric_with_pos_label(y_true, y_pred, *, pos_label):
        captured["pos_label"] = pos_label
        return 0.42

    m = Metric(
        name="pos_metric",
        function=metric_with_pos_label,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    rows = m.rows(report=binary_classification_report, data_source="test")
    assert rows[0]["score"] == 0.42
    assert captured["pos_label"] == binary_classification_report.pos_label


def test_metric_rows_dict_return_classification(binary_classification_report):
    def per_class(y_true, y_pred) -> dict:
        return {0: 0.5, 1: 0.7}

    def scorer(est, X, y_true):
        return per_class(y_true, est.predict(X))

    m = Metric(
        name="per_class",
        function=scorer,
        greater_is_better=True,
        function_kind=FunctionKind.SCORER,
    )
    rows = m.rows(report=binary_classification_report, data_source="test")
    assert {row["metric_verbose_name"]: row["score"] for row in rows} == {
        0: 0.5,
        1: 0.7,
    }


def test_metric_new_callable():
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
    assert metric.function_kind == FunctionKind.SCORER


def test_metric_new_callable_with_name():
    metric = Metric.new(
        business_loss_scorer, name="my_loss", kwargs={"cost_fp": 10, "cost_fn": 5}
    )

    assert metric.name == "my_loss"
    assert metric.verbose_name == "My Loss"
    assert metric.function is business_loss_scorer
    assert metric.kwargs == {"cost_fp": 10, "cost_fn": 5}


def test_metric_new_callable_missing_kwargs():
    err_msg = re.escape(
        "Callable 'business_loss_scorer' has required parameter(s) "
        "('cost_fp', 'cost_fn') not covered by the provided kwargs."
    )
    with pytest.raises(MissingKwargsError, match=err_msg):
        Metric.new(business_loss_scorer)


def test_metric_new_callable_metric_y():
    err_msg = re.escape(
        "Expected a scorer callable with an estimator as its first argument; "
        "got first argument 'y_true'"
    )
    with pytest.raises(TypeError, match=err_msg):
        Metric.new(business_loss_metric)


def test_metric_new_callable_not_enough_positional_args():
    def metric(true_labels, predicted_labels, *, some_kwarg):
        pass

    err_msg = re.escape(
        "Expected a scorer callable with at least 3 positional arguments "
        "(estimator, X, y); got ['true_labels', 'predicted_labels']"
    )
    with pytest.raises(TypeError, match=err_msg):
        Metric.new(metric)


def test_metric_new_scorer():
    scorer = make_scorer(accuracy_score, response_method="predict")
    metric = Metric.new(scorer)

    assert isinstance(metric, Metric)
    assert metric.name == "accuracy_score"
    assert metric.function is accuracy_score
    assert metric.greater_is_better is True
    assert metric.response_method == "predict"
    assert metric.function_kind == FunctionKind.METRIC


def test_metric_new_sklearn_scorer_negative():
    scorer = get_scorer("neg_mean_squared_error")
    result = Metric.new(scorer)
    assert result.name == "mean_squared_error"
    assert result.greater_is_better is False


def test_metric_new_metric_passthrough():
    original = Metric(name="original", function=get_scorer("accuracy"))
    result = Metric.new(original)

    assert result.name == "original"
    assert result is original


def test_metric_new_metric_with_name():
    original = Metric(name="original", function=get_scorer("accuracy"))
    result = Metric.new(original, name="renamed")

    assert result.name == "renamed"
    assert result.verbose_name == "Renamed"
    assert original.name == "original"


def test_metric_new_string():
    metric = Metric.new("f1")

    assert isinstance(metric, Metric)
    assert metric.name == "f1"
    assert metric.function is f1_score


def test_metric_new_string_alias_resolved():
    result = Metric.new("mean_squared_error")
    assert result.name == "mean_squared_error"
    assert result.greater_is_better is False


def test_metric_new_string_with_neg_prefix_keeps_name():
    """String names with ``neg_`` keep the user-provided name."""
    result = Metric.new("neg_mean_squared_error")
    assert result.name == "neg_mean_squared_error"
    assert result.greater_is_better is False


def test_metric_new_invalid_string():
    with pytest.raises(ValueError, match="Invalid metric"):
        Metric.new("xyz")


def test_metric_new_invalid_type():
    with pytest.raises(TypeError, match="Cannot create"):
        Metric.new(42)


def test_metric_new_functools_partial():
    partial_func = functools.partial(business_loss_scorer, cost_fp=10, cost_fn=5)
    metric = Metric.new(partial_func)

    assert metric.name == "business_loss_scorer"
    assert metric.function is partial_func


def test_metric_new_callable_object_without_name():
    class MyScorer:
        def __call__(self, estimator, X, y):
            return get_scorer("accuracy")(estimator, X, y)

    metric = Metric.new(MyScorer())

    assert metric.name == "MyScorer"


@pytest.mark.parametrize("friendly", list(_METRIC_ALIASES))
def test_metric_alias_via_metric_new(friendly):
    result = Metric.new(friendly)
    assert result.name == friendly
    assert result.greater_is_better is False


def test_metric_registry_binary_classification_filters(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    names = list(registry.keys())
    assert "accuracy" in names
    assert "precision" in names
    assert "recall" in names
    assert "roc_auc" in names
    assert "log_loss" in names
    assert "brier_score" in names
    assert "fit_time" in names
    assert "predict_time" in names
    assert "score" in names
    assert "r2" not in names
    assert "rmse" not in names
    assert "precision_macro" not in names


def test_metric_registry_regression_filters(regression_report):
    registry = MetricRegistry(regression_report)
    names = list(registry.keys())
    assert "r2" in names
    assert "rmse" in names
    assert "mae" in names
    assert "mape" in names
    assert "fit_time" in names
    assert "predict_time" in names
    assert "accuracy" not in names
    assert "precision" not in names


def test_metric_registry_no_proba_classifier_filters(
    classifier_no_predict_proba_report,
):
    registry = MetricRegistry(classifier_no_predict_proba_report)
    names = list(registry.keys())
    assert "accuracy" in names
    assert "precision" in names
    assert "recall" in names
    assert "roc_auc" not in names
    assert "log_loss" not in names
    assert "brier_score" not in names


def test_metric_registry_iteration_order(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    default_order = [
        m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
    ]
    # Score is inserted at the front when available
    expected = (
        ["score", *default_order]
        if Score.available(binary_classification_report)
        else default_order
    )
    assert list(registry.keys()) == expected


def test_metric_registry_repr(binary_classification_report):
    registry = binary_classification_report._metric_registry
    result = repr(registry)
    assert result.startswith("MetricRegistry")
    assert "accuracy" in result


def test_metric_registry_add(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    m = Metric(
        name="custom",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    registry.add(m)
    assert "custom" in registry


def test_metric_registry_add_position_first(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="custom_a", function=None), position="first")
    assert next(iter(registry.keys())) == "custom_a"


def test_metric_registry_add_position_last(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="custom_z", function=None), position="last")
    assert tuple(registry.keys())[-1] == "custom_z"


def test_metric_registry_add_multiple_first_lifo(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="a", function=None), position="first")
    registry.add(Metric(name="b", function=None), position="first")
    keys = list(registry.keys())
    assert keys[0] == "b"
    assert keys[1] == "a"


def test_metric_registry_add_multiple_last_fifo(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="m_a", function=None), position="last")
    registry.add(Metric(name="m_b", function=None), position="last")
    keys = list(registry.keys())
    assert keys[-2] == "m_a"
    assert keys[-1] == "m_b"


def test_metric_registry_add_invalid_position(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    m = Metric(
        name="only_for_position_test",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    with pytest.raises(ValueError, match="position must be 'first' or 'last'"):
        registry.add(m, position="middle")  # type: ignore[arg-type]


def test_metric_registry_readd_default_metric(binary_classification_report):
    """Default metrics may be removed and re-added under the same name."""
    registry = MetricRegistry(binary_classification_report)
    assert "accuracy" in registry
    registry.remove(report=binary_classification_report, name="accuracy")

    registry.add(
        Metric(
            name="accuracy",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
    )
    assert "accuracy" in registry


def test_metric_registry_add_score_name_reserved(binary_classification_report):
    """``"score"`` stays reserved even once removed from the registry."""
    assert RESERVED_METRIC_NAMES == frozenset({Score.name})

    report = binary_classification_report
    report.metrics.remove("score")
    assert "score" not in report._metric_registry

    with pytest.raises(
        ValueError, match="Cannot add 'score': it is a reserved name."
    ):
        report.metrics.add(
            make_scorer(accuracy_score, response_method="predict"), name="score"
        )


def test_metric_registry_add_duplicate_raises(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="dup", function=None))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot add 'dup': it already exists. "
            "Remove it first using the `remove` method."
        ),
    ):
        registry.add(Metric(name="dup", function=None))


def test_metric_registry_remove_custom(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    registry.add(Metric(name="custom", function=None))
    assert "custom" in registry
    registry.remove(report=binary_classification_report, name="custom")
    assert "custom" not in registry


def test_metric_registry_remove_default(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    assert "accuracy" in registry
    registry.remove(report=binary_classification_report, name="accuracy")
    assert "accuracy" not in registry


def test_metric_registry_remove_unknown_raises(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    with pytest.raises(KeyError) as exc_info:
        registry.remove(report=binary_classification_report, name="no_such_metric")
    assert exc_info.value.args[0] == "no_such_metric"


def test_metric_registry_remove_clears_only_target_cache_entries(
    binary_classification_report,
):
    registry = MetricRegistry(binary_classification_report)
    report = binary_classification_report

    registry.add(Metric(name="metric_a", function=None))
    registry.add(Metric(name="metric_b", function=None))

    a_on_test = make_cache_key("metrics", "test", "metric_a", {})
    a_on_train = make_cache_key("metrics", "train", "metric_a", {})
    b_on_test = make_cache_key("metrics", "test", "metric_b", {})
    report._cache[a_on_test] = 0.1
    report._cache[a_on_train] = 0.2
    report._cache[b_on_test] = 0.3

    registry.remove(report=report, name="metric_a")

    assert a_on_test not in report._cache
    assert a_on_train not in report._cache
    assert b_on_test in report._cache


def test_metric_registry_works_with_test_only_report(
    binary_classification_train_test_split,
):
    _, X_test, _, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        LogisticRegression().fit(X_test, y_test), X_test=X_test, y_test=y_test
    )
    registry = MetricRegistry(report)
    assert "accuracy" in registry


def test_metric_registry_default_contents_match_seed(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    available_defaults = {
        m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
    }
    if Score.available(binary_classification_report):
        available_defaults.add("score")
    assert set(registry.keys()) == available_defaults
