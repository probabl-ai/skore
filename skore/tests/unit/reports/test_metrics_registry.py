"""Unit tests for Metric, MetricRegistry, and related helpers."""

from __future__ import annotations

import functools
import pickle
import re

import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
)
from sklearn.metrics._scorer import _BaseScorer

from skore import EstimatorReport
from skore._sklearn.metrics import (
    _METRIC_ALIASES,
    BUILTIN_METRICS,
    R2,
    Accuracy,
    Brier,
    FitTime,
    FunctionKind,
    LogLoss,
    Mae,
    Mape,
    Metric,
    MetricRegistry,
    MissingKwargsError,
    Precision,
    PrecisionMacro,
    PredictTime,
    Recall,
    RecallMacro,
    Rmse,
    RocAuc,
    RocAucMacro,
    Score,
)
from skore._utils._cache_key import make_cache_key


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
def multiclass_classification_report(
    logistic_multiclass_classification_with_train_test,
):
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
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


@pytest.fixture
def multioutput_regression_report(linear_regression_multioutput_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def svc_binary_classification_report(svc_binary_classification_with_train_test):
    """SVC binary report: has decision_function but no predict_proba."""
    estimator, X_train, X_test, y_train, y_test = (
        svc_binary_classification_with_train_test
    )
    return EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def classifier_no_predict_proba_report(
    custom_classifier_no_predict_proba_with_test,
):
    """Custom classifier without predict_proba and decision_function."""
    estimator, X_test, y_test = custom_classifier_no_predict_proba_with_test
    return EstimatorReport(estimator, X_test=X_test, y_test=y_test)


def business_loss_metric(y_true, y_pred, *, cost_fp, cost_fn):
    """Custom (y_true, y_pred) metric used to test the y-prefix guard in Metric.new."""
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn


def business_loss_scorer(estimator, X, y, cost_fp, cost_fn):
    """Custom (estimator, X, y) scorer with required kwargs."""
    y_pred = estimator.predict(X)
    return business_loss_metric(y, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)


def test_function_kind_members():
    assert {member.name for member in FunctionKind} == {"METRIC", "SCORER"}


def test_function_kind_members_distinct():
    assert FunctionKind.METRIC is not FunctionKind.SCORER
    assert FunctionKind.METRIC.value != FunctionKind.SCORER.value


def test_missing_kwargs_error_attributes_and_message():
    err = MissingKwargsError(business_loss_scorer, ("cost_fp", "cost_fn"))
    assert err.metric == "business_loss_scorer"
    assert err.missing_kwargs == ("cost_fp", "cost_fn")
    assert err.msg == (
        "Callable 'business_loss_scorer' has required parameter(s) "
        "('cost_fp', 'cost_fn') not covered by the provided kwargs."
    )


def test_missing_kwargs_error_str():
    err = MissingKwargsError(business_loss_scorer, ["cost_fp"])
    assert str(err) == err.msg


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


def test_metric_init_subclass_path_only_sets_kwargs():
    """When `name=None`, only ``kwargs`` is set as an instance attribute."""
    m = Metric()
    assert m.kwargs == {}
    assert "name" not in m.__dict__
    assert "verbose_name" not in m.__dict__
    assert "greater_is_better" not in m.__dict__


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


def test_metric_aliases_resolve_via_get_scorer():
    scorer_names = set(sklearn.metrics.get_scorer_names())
    for friendly, neg_form in _METRIC_ALIASES.items():
        if neg_form in scorer_names:
            assert sklearn.metrics.get_scorer(neg_form) is not None
            if friendly in scorer_names:
                assert sklearn.metrics.get_scorer(friendly) is not None
            else:
                assert friendly not in scorer_names
        elif friendly in scorer_names:
            assert sklearn.metrics.get_scorer(friendly) is not None
        else:
            pytest.fail(
                f"Alias {friendly!r} -> {neg_form!r}: neither string is registered "
                f"in sklearn.metrics.get_scorer_names() for sklearn "
                f"{sklearn.__version__}."
            )


@pytest.mark.parametrize("friendly", list(_METRIC_ALIASES))
def test_metric_alias_via_metric_new(friendly):
    result = Metric.new(friendly)
    assert result.name == friendly
    assert result.greater_is_better is False


_BUILTIN_ATTRS = [
    (Accuracy, "accuracy", "Accuracy", True, FunctionKind.METRIC),
    (Precision, "precision", "Precision", True, FunctionKind.METRIC),
    (Recall, "recall", "Recall", True, FunctionKind.METRIC),
    (Brier, "brier_score", "Brier score", False, FunctionKind.METRIC),
    (RocAuc, "roc_auc", "ROC AUC", True, FunctionKind.METRIC),
    (LogLoss, "log_loss", "Log loss", False, FunctionKind.METRIC),
    (R2, "r2", "R²", True, FunctionKind.METRIC),
    (Rmse, "rmse", "RMSE", False, FunctionKind.METRIC),
    (Mae, "mae", "MAE", False, FunctionKind.METRIC),
    (Mape, "mape", "MAPE", False, FunctionKind.METRIC),
    (FitTime, "fit_time", "Fit time (s)", False, None),
    (PredictTime, "predict_time", "Predict time (s)", False, None),
]


@pytest.mark.parametrize(
    ("cls", "name", "verbose", "greater_is_better", "function_kind"),
    _BUILTIN_ATTRS,
)
def test_builtin_class_attributes(cls, name, verbose, greater_is_better, function_kind):
    assert cls.name == name
    assert cls.verbose_name == verbose
    assert cls.greater_is_better is greater_is_better
    assert cls.function_kind is function_kind


def test_accuracy_available_for_classification(
    binary_classification_report,
    multiclass_classification_report,
    regression_report,
):
    assert Accuracy().available(binary_classification_report)
    assert Accuracy().available(multiclass_classification_report)
    assert not Accuracy().available(regression_report)


def test_precision_recall_available_for_classification(
    binary_classification_report,
    multiclass_classification_report,
    regression_report,
):
    for cls in (Precision, Recall):
        assert cls().available(binary_classification_report)
        assert cls().available(multiclass_classification_report)
        assert not cls().available(regression_report)


def test_macro_metrics_only_multiclass(
    binary_classification_report,
    multiclass_classification_report,
):
    for cls in (PrecisionMacro, RecallMacro, RocAucMacro):
        assert not cls().available(binary_classification_report)
        assert cls().available(multiclass_classification_report)


def test_brier_only_binary_with_predict_proba(
    binary_classification_report,
    multiclass_classification_report,
    regression_report,
    svc_binary_classification_report,
):
    assert Brier().available(binary_classification_report)
    assert not Brier().available(svc_binary_classification_report)
    assert not Brier().available(multiclass_classification_report)
    assert not Brier().available(regression_report)


def test_roc_auc_availability(
    binary_classification_report,
    multiclass_classification_report,
    regression_report,
    svc_binary_classification_report,
    classifier_no_predict_proba_report,
):
    assert RocAuc().available(binary_classification_report)
    assert RocAuc().available(svc_binary_classification_report)
    assert RocAuc().available(multiclass_classification_report)
    assert not RocAuc().available(classifier_no_predict_proba_report)
    assert not RocAuc().available(regression_report)


def test_log_loss_classification_with_predict_proba(
    binary_classification_report,
    multiclass_classification_report,
    regression_report,
    svc_binary_classification_report,
):
    assert LogLoss().available(binary_classification_report)
    assert LogLoss().available(multiclass_classification_report)
    assert not LogLoss().available(svc_binary_classification_report)
    assert not LogLoss().available(regression_report)


def test_regression_metrics_only_for_regression(
    binary_classification_report,
    regression_report,
    multioutput_regression_report,
):
    for cls in (R2, Rmse, Mae, Mape):
        assert cls().available(regression_report)
        assert cls().available(multioutput_regression_report)
        assert not cls().available(binary_classification_report)


def test_fit_time_predict_time_always_available(
    binary_classification_report, regression_report
):
    assert FitTime().available(binary_classification_report)
    assert FitTime().available(regression_report)
    assert PredictTime().available(binary_classification_report)
    assert PredictTime().available(regression_report)


def test_fit_time_rows(binary_classification_report):
    rows = FitTime().rows(report=binary_classification_report, data_source="test")
    assert len(rows) == 1
    assert np.isnan(rows[0]["score"])


def test_fit_time_rows_cast_false(binary_classification_report):
    assert binary_classification_report._fit_time is None
    rows = FitTime().rows(
        report=binary_classification_report, data_source="test", cast=False
    )
    assert rows[0]["score"] is None


def test_predict_time_rows_no_cache(binary_classification_report):
    """With no cached predict time, ``cast=True`` returns ``nan``."""
    binary_classification_report._predict_time.clear()
    rows = PredictTime().rows(report=binary_classification_report, data_source="test")
    assert np.isnan(rows[0]["score"])


def test_predict_time_rows_cast_false(binary_classification_report):
    """With no cached predict time, ``cast=False`` returns ``None``."""
    binary_classification_report._predict_time.clear()
    rows = PredictTime().rows(
        report=binary_classification_report, data_source="test", cast=False
    )
    assert rows[0]["score"] is None


def test_accuracy_pretty(binary_classification_report):
    score = Accuracy().pretty(report=binary_classification_report, data_source="test")
    assert 0.0 <= score <= 1.0


def test_builtin_metrics_contains_expected_classes():
    types = [type(m) for m in BUILTIN_METRICS]
    assert types == [
        Accuracy,
        Precision,
        PrecisionMacro,
        Recall,
        RecallMacro,
        RocAuc,
        RocAucMacro,
        LogLoss,
        Brier,
        R2,
        Rmse,
        Mae,
        Mape,
        FitTime,
        PredictTime,
    ]


def test_builtin_metrics_unique_names():
    names = [m.name for m in BUILTIN_METRICS]
    assert len(names) == len(set(names))


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


def test_metric_registry_no_proba_classifier_filters(classifier_no_predict_proba_report):
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
    builtin_order = [
        m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
    ]
    # Score is inserted at the front when available
    expected = ["score", *builtin_order] if Score.available(binary_classification_report) else builtin_order
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


def test_metric_registry_add_builtin_name_conflict(binary_classification_report):
    registry = MetricRegistry(binary_classification_report)
    m = Metric(
        name="accuracy",
        function=accuracy_score,
        response_method="predict",
        greater_is_better=True,
        function_kind=FunctionKind.METRIC,
    )
    with pytest.raises(
        ValueError, match="Cannot add 'accuracy': it is a built-in metric name."
    ):
        registry.add(m)


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


def test_metric_registry_remove_builtin(binary_classification_report):
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

    report._cache[make_cache_key("metrics", "test", "metric_a", {})] = 0.1
    report._cache[make_cache_key("metrics", "train", "metric_a", {})] = 0.2
    report._cache[make_cache_key("metrics", "test", "metric_b", {})] = 0.3

    registry.remove(report=report, name="metric_a")

    assert not any(k[2] == "metric_a" for k in report._cache)
    assert any(k[2] == "metric_b" for k in report._cache)


def test_metric_registry_works_with_test_only_report(
    binary_classification_train_test_split,
):
    _, X_test, _, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        LogisticRegression().fit(X_test, y_test), X_test=X_test, y_test=y_test
    )
    registry = MetricRegistry(report)
    assert "accuracy" in registry


def test_metric_registry_default_metrics_match_builtin_metrics(
    binary_classification_report,
):
    registry = MetricRegistry(binary_classification_report)
    available_builtins = {
        m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
    }
    if Score.available(binary_classification_report):
        available_builtins.add("score")
    assert set(registry.keys()) == available_builtins


def test_sklearn_scorer_protocol_recognises_basescorer():
    scorer = make_scorer(accuracy_score, response_method="predict")
    assert isinstance(scorer, _BaseScorer)
    assert hasattr(scorer, "_score_func")
    assert hasattr(scorer, "_response_method")
    assert hasattr(scorer, "_kwargs")
