"""Tests for unit tests for metrics registry functionality."""

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
    PredictTime,
    Recall,
    Rmse,
    RocAuc,
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


class TestFunctionKind:
    """Tests for the FunctionKind enum."""

    def test_members(self):
        assert {member.name for member in FunctionKind} == {"METRIC", "SCORER"}

    def test_members_distinct(self):
        assert FunctionKind.METRIC is not FunctionKind.SCORER
        assert FunctionKind.METRIC.value != FunctionKind.SCORER.value


class TestMissingKwargsError:
    """Tests for MissingKwargsError."""

    def test_attributes_and_message(self):
        err = MissingKwargsError(business_loss_scorer, ("cost_fp", "cost_fn"))
        assert err.metric == "business_loss_scorer"
        assert err.missing_kwargs == ("cost_fp", "cost_fn")
        assert err.msg == (
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
        )

    def test_str(self):
        err = MissingKwargsError(business_loss_scorer, ["cost_fp"])
        assert str(err) == err.msg

    def test_partial_callable_name(self):
        partial = functools.partial(business_loss_scorer, cost_fp=1)
        err = MissingKwargsError(partial, ("cost_fn",))
        # `_callable_name` falls back to underlying ``func`` for ``functools.partial``.
        assert err.metric == "business_loss_scorer"


class TestMetricInit:
    """Tests for Metric.__init__."""

    def test_full_args(self):
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

    def test_default_verbose_name(self):
        m = Metric(name="my_metric")
        assert m.verbose_name == "My Metric"

    def test_kwargs_default(self):
        m = Metric(name="x")
        assert m.kwargs == {}

    def test_subclass_path_only_sets_kwargs(self):
        """When `name=None`, only ``kwargs`` is set as an instance attribute.

        This is the path taken by built-in subclasses where ``name`` and other
        fields are class attributes.
        """
        m = Metric()
        assert m.kwargs == {}
        # name/verbose_name/etc. were not assigned as instance attrs
        assert "name" not in m.__dict__
        assert "verbose_name" not in m.__dict__
        assert "greater_is_better" not in m.__dict__

    def test_subclass_path_kwargs_propagated(self):
        m = Metric(kwargs={"average": "macro"})
        assert m.kwargs == {"average": "macro"}

    def test_greater_is_better_none(self):
        m = Metric(name="t", greater_is_better=None)
        assert m.greater_is_better is None
        m2 = Metric(name="t", greater_is_better=True)
        assert m2.greater_is_better is True


class TestMetricGetState:
    """Tests for Metric.__getstate__ (used during pickling)."""

    def test_drops_lambda(self):
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

    def test_keeps_picklable(self):
        m = Metric(
            name="keep",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
        state = m.__getstate__()
        assert state["function"] is accuracy_score

    def test_round_trip(self):
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


class TestMetricRepr:
    """Tests for Metric.__repr__."""

    def test_repr_default(self):
        m = Metric(name="accuracy", function=None, greater_is_better=True)
        assert repr(m) == (
            "Metric(name='accuracy', verbose_name='Accuracy', function=None, "
            "greater_is_better=True, response_method=None, kwargs={})"
        )

    def test_repr_with_kwargs(self):
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


class TestMetricAvailable:
    """Tests for Metric.available()."""

    def test_default_returns_true(self, binary_classification_report):
        m = Metric(name="anything")
        assert m.available(binary_classification_report) is True


class TestMetricCall:
    """Tests for Metric.__call__."""

    def test_no_function_raises(self, binary_classification_report):
        m = Metric(name="abstract_metric", function=None)
        with pytest.raises(
            ValueError, match="Metric 'abstract_metric' has no scoring function."
        ):
            m(report=binary_classification_report)

    def test_function_kind_metric(self, binary_classification_report):
        """METRIC kind: calls function(y_true, y_pred, **kwargs)."""
        m = Metric(
            name="acc",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
        score = m(report=binary_classification_report)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_function_kind_scorer(self, binary_classification_report):
        """SCORER kind: calls function(estimator, X, y, **kwargs)."""

        def my_scorer(estimator, X, y_true):
            return float((estimator.predict(X) == y_true).mean())

        m = Metric(
            name="scorer_acc",
            function=my_scorer,
            greater_is_better=True,
            function_kind=FunctionKind.SCORER,
        )
        score = m(report=binary_classification_report)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_cache_hit(self, binary_classification_report):
        """Cache hit returns the cached value without recomputing."""
        m = Metric(
            name="acc",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
        sentinel = object()
        cache_key = make_cache_key("test", "acc", {})
        binary_classification_report._cache[cache_key] = sentinel
        assert m(report=binary_classification_report) is sentinel

    def test_cache_populated(self, binary_classification_report):
        m = Metric(
            name="acc",
            function=accuracy_score,
            response_method="predict",
            greater_is_better=True,
            function_kind=FunctionKind.METRIC,
        )
        score = m(report=binary_classification_report)
        cached = binary_classification_report._cache[make_cache_key("test", "acc", {})]
        assert cached == score

    def test_pos_label_injection(self, binary_classification_report):
        """When the function signature has `pos_label`, the report's `pos_label` is
        injected automatically."""
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
        score = m(report=binary_classification_report)
        assert score == 0.42
        assert captured["pos_label"] == binary_classification_report.pos_label

    def test_dict_return_classification(self, binary_classification_report):
        """A dict-returning metric stays a dict for classification reports."""

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
        score = m(report=binary_classification_report)
        assert score == {0: 0.5, 1: 0.7}

    def test_list_of_one_collapses(self, regression_report):
        """A list of length 1 collapses to its single scalar element."""

        def scorer(est, X, y):
            return [3.14]

        m = Metric(
            name="single_list",
            function=scorer,
            greater_is_better=True,
            function_kind=FunctionKind.SCORER,
        )
        assert m(report=regression_report) == 3.14

    def test_numpy_array_converts_via_item(self, regression_report):
        """A 0-d numpy array is converted via .item()."""

        def scorer(est, X, y):
            return np.float64(1.5)

        m = Metric(
            name="np_scalar",
            function=scorer,
            greater_is_better=True,
            function_kind=FunctionKind.SCORER,
        )
        result = m(report=regression_report)
        assert result == 1.5
        assert isinstance(result, float)

    def test_numpy_array_converts_via_tolist(self, multiclass_classification_report):
        """A 1-d numpy array of classification per-class scores is converted to dict."""

        def scorer(est, X, y):
            return np.array([0.5, 0.6, 0.7])

        m = Metric(
            name="per_class_array",
            function=scorer,
            greater_is_better=True,
            function_kind=FunctionKind.SCORER,
        )
        result = m(report=multiclass_classification_report)
        assert isinstance(result, dict)
        # The keys correspond to the ``classes_`` of the estimator
        classes = multiclass_classification_report._estimator.classes_.tolist()
        assert set(result.keys()) == set(classes)

    def test_default_kwargs_merged_with_call_kwargs(self, binary_classification_report):
        """Default kwargs are merged with call-time kwargs (call-time wins)."""
        captured = {}

        def my_scorer(est, X, y, *, factor=1):
            captured["factor"] = factor
            return float(factor)

        m = Metric(
            name="merge",
            function=my_scorer,
            greater_is_better=True,
            function_kind=FunctionKind.SCORER,
            kwargs={"factor": 2},
        )
        # default kwargs flow in
        m(report=binary_classification_report)
        assert captured["factor"] == 2

        # call-time kwargs override
        captured.clear()
        # use a different cache key by passing kwargs
        m(report=binary_classification_report, factor=3)
        assert captured["factor"] == 3


class TestMetricNew:
    """Tests for Metric.new (covers all branches)."""

    def test_metric_instance_passthrough(self):
        original = Metric(name="o", function=accuracy_score)
        result = Metric.new(original)
        assert result is original

    def test_metric_instance_with_name_renames(self):
        original = Metric(name="o", function=accuracy_score)
        result = Metric.new(original, name="renamed")
        assert result is not original
        assert result.name == "renamed"
        assert result.verbose_name == "Renamed"
        assert original.name == "o"

    def test_sklearn_scorer(self):
        scorer = make_scorer(accuracy_score, response_method="predict")
        result = Metric.new(scorer)
        assert isinstance(result, Metric)
        assert result.name == "accuracy_score"
        assert result.function is accuracy_score
        assert result.greater_is_better is True
        assert result.response_method == "predict"
        assert result.function_kind == FunctionKind.METRIC

    def test_sklearn_scorer_negative(self):
        scorer = get_scorer("neg_mean_squared_error")
        result = Metric.new(scorer)
        assert isinstance(result, Metric)
        assert result.greater_is_better is False

    def test_string_name(self):
        result = Metric.new("f1")
        assert isinstance(result, Metric)
        assert result.name == "f1"
        assert result.function is f1_score

    def test_string_name_neg_prefix_stripped(self):
        result = Metric.new("neg_mean_squared_error")
        assert result.name == "mean_squared_error"
        assert result.greater_is_better is False

    def test_string_name_alias_resolved(self):
        """Aliases without ``neg_`` prefix are resolved automatically."""
        result = Metric.new("mean_squared_error")
        assert result.name == "mean_squared_error"
        assert result.greater_is_better is False

    def test_string_name_invalid(self):
        with pytest.raises(ValueError, match="Invalid metric: 'xyz'"):
            Metric.new("xyz")

    def test_callable_with_kwargs(self):
        result = Metric.new(
            business_loss_scorer,
            greater_is_better=False,
            kwargs={"cost_fp": 10, "cost_fn": 5},
        )
        assert isinstance(result, Metric)
        assert result.name == "business_loss_scorer"
        assert result.function is business_loss_scorer
        assert result.greater_is_better is False
        assert result.kwargs == {"cost_fp": 10, "cost_fn": 5}
        assert result.function_kind == FunctionKind.SCORER

    def test_callable_with_custom_name(self):
        result = Metric.new(
            business_loss_scorer,
            name="my_loss",
            kwargs={"cost_fp": 10, "cost_fn": 5},
        )
        assert result.name == "my_loss"
        assert result.verbose_name == "My Loss"

    def test_callable_missing_kwargs(self):
        err_msg = re.escape(
            "Callable 'business_loss_scorer' has required parameter(s) "
            "('cost_fp', 'cost_fn') not covered by the provided kwargs."
        )
        with pytest.raises(MissingKwargsError, match=err_msg):
            Metric.new(business_loss_scorer)

    def test_callable_first_arg_is_y(self):
        """A metric of the form ``(y_true, y_pred, **kw)`` is rejected."""
        err_msg = re.escape(
            "Expected a scorer callable with an estimator as its first argument; "
            "got first argument 'y_true'"
        )
        with pytest.raises(TypeError, match=err_msg):
            Metric.new(business_loss_metric)

    def test_callable_not_enough_positional_args(self):
        def metric(true_labels, predicted_labels, *, some_kwarg):
            pass

        err_msg = re.escape(
            "Expected a scorer callable with at least 3 positional arguments "
            "(estimator, X, y); got ['true_labels', 'predicted_labels']"
        )
        with pytest.raises(TypeError, match=err_msg):
            Metric.new(metric)

    def test_functools_partial(self):
        partial_func = functools.partial(business_loss_scorer, cost_fp=10, cost_fn=5)
        result = Metric.new(partial_func)
        assert result.name == "business_loss_scorer"
        assert result.function is partial_func

    def test_callable_object_without_name(self):
        class MyScorer:
            def __call__(self, estimator, X, y):
                return 1.0

        result = Metric.new(MyScorer())
        assert result.name == "MyScorer"

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Cannot create"):
            Metric.new(42)


class TestMetricAliases:
    """The aliases table maps user-friendly names to the sklearn `neg_*` form."""

    def test_aliases_resolve_via_get_scorer(self):
        for friendly, neg_form in _METRIC_ALIASES.items():
            assert sklearn.metrics.get_scorer(neg_form) is not None
            # The friendly name itself must NOT be a registered scorer name
            assert friendly not in sklearn.metrics.get_scorer_names()

    @pytest.mark.parametrize("friendly", list(_METRIC_ALIASES))
    def test_alias_via_metric_new(self, friendly):
        """``Metric.new(<friendly>)`` resolves the alias and uses the friendly name."""
        result = Metric.new(friendly)
        assert result.name == friendly
        assert result.greater_is_better is False


# Expected class-level attributes for built-ins, in (cls, name, verbose, gib,
# function_kind) form.
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
    """Class attributes match the contract used by the registry."""
    assert cls.name == name
    assert cls.verbose_name == verbose
    assert cls.greater_is_better is greater_is_better
    assert cls.function_kind is function_kind


class TestBuiltinAvailability:
    """Each built-in's ``available()`` predicate against representative reports."""

    def test_accuracy_available_for_classification(
        self,
        binary_classification_report,
        multiclass_classification_report,
        regression_report,
    ):
        assert Accuracy().available(binary_classification_report)
        assert Accuracy().available(multiclass_classification_report)
        assert not Accuracy().available(regression_report)

    def test_precision_recall_available_for_classification(
        self,
        binary_classification_report,
        multiclass_classification_report,
        regression_report,
    ):
        for cls in (Precision, Recall):
            assert cls().available(binary_classification_report)
            assert cls().available(multiclass_classification_report)
            assert not cls().available(regression_report)

    def test_brier_only_binary_with_predict_proba(
        self,
        binary_classification_report,
        multiclass_classification_report,
        regression_report,
        svc_binary_classification_report,
    ):
        assert Brier().available(binary_classification_report)
        # SVC lacks predict_proba
        assert not Brier().available(svc_binary_classification_report)
        assert not Brier().available(multiclass_classification_report)
        assert not Brier().available(regression_report)

    def test_roc_auc_availability(
        self,
        binary_classification_report,
        multiclass_classification_report,
        regression_report,
        svc_binary_classification_report,
        classifier_no_predict_proba_report,
    ):
        # binary: predict_proba OR decision_function works
        assert RocAuc().available(binary_classification_report)
        assert RocAuc().available(svc_binary_classification_report)
        # multiclass: requires predict_proba
        assert RocAuc().available(multiclass_classification_report)
        # neither predict_proba nor decision_function
        assert not RocAuc().available(classifier_no_predict_proba_report)
        # not available for regression
        assert not RocAuc().available(regression_report)

    def test_log_loss_classification_with_predict_proba(
        self,
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
        self,
        binary_classification_report,
        regression_report,
        multioutput_regression_report,
    ):
        for cls in (R2, Rmse, Mae, Mape):
            assert cls().available(regression_report)
            assert cls().available(multioutput_regression_report)
            assert not cls().available(binary_classification_report)

    def test_fit_time_predict_time_always_available(
        self, binary_classification_report, regression_report
    ):
        assert FitTime().available(binary_classification_report)
        assert FitTime().available(regression_report)
        assert PredictTime().available(binary_classification_report)
        assert PredictTime().available(regression_report)


class TestBuiltinCall:
    """Spot-check that the built-in __call__ overrides work in isolation."""

    def test_fit_time_returns_float(self, binary_classification_report):
        ft = FitTime()
        assert isinstance(ft(report=binary_classification_report), float)

    def test_fit_time_no_cast_when_unfitted_is_none(self, binary_classification_report):
        """When the estimator was already fitted, ``fit_time_`` is ``None`` and
        ``cast=False`` exposes that ``None`` rather than ``nan``.
        """
        ft = FitTime()
        # The fixture's estimator was pre-fitted, so fit_time_ is None.
        assert binary_classification_report.fit_time_ is None
        assert ft(report=binary_classification_report, cast=False) is None

    def test_fit_time_unfitted_returns_nan(self, binary_classification_report):
        """``cast=True`` (default) turns ``None`` into ``nan``."""
        assert np.isnan(FitTime()(report=binary_classification_report))

    def test_predict_time_no_cache_returns_nan(self, binary_classification_report):
        pt = PredictTime()
        # Without computing predictions first, predict_time is not cached
        result = pt(report=binary_classification_report)
        assert np.isnan(result)

    def test_predict_time_no_cast(self, binary_classification_report):
        pt = PredictTime()
        assert pt(report=binary_classification_report, cast=False) is None

    def test_accuracy_call(self, binary_classification_report):
        score = Accuracy()(report=binary_classification_report)
        assert 0.0 <= score <= 1.0

    def test_precision_call_binary(self, binary_classification_report):
        # pos_label is set to 1 on the fixture; the override sets average="binary"
        score = Precision()(report=binary_classification_report)
        assert 0.0 <= score <= 1.0

    def test_recall_call_binary(self, binary_classification_report):
        score = Recall()(report=binary_classification_report)
        assert 0.0 <= score <= 1.0

    def test_brier_call(self, binary_classification_report):
        score = Brier()(report=binary_classification_report)
        # Brier lives in [0, 1]
        assert 0.0 <= score <= 1.0

    def test_roc_auc_call(self, binary_classification_report):
        score = RocAuc()(report=binary_classification_report)
        assert 0.0 <= score <= 1.0

    def test_log_loss_call(self, binary_classification_report):
        score = LogLoss()(report=binary_classification_report)
        assert score >= 0.0

    def test_regression_calls(self, regression_report):
        for cls in (R2, Rmse, Mae, Mape):
            score = cls()(report=regression_report)
            assert isinstance(score, float)


class TestBuiltinMetricsList:
    """Tests for the BUILTIN_METRICS list."""

    def test_contains_all_classes(self):
        types = [type(m) for m in BUILTIN_METRICS]
        assert types == [
            Accuracy,
            Precision,
            Recall,
            RocAuc,
            LogLoss,
            Brier,
            R2,
            Rmse,
            Mae,
            Mape,
            FitTime,
            PredictTime,
        ]

    def test_unique_names(self):
        names = [m.name for m in BUILTIN_METRICS]
        assert len(names) == len(set(names))


class TestMetricRegistryInit:
    """Tests for MetricRegistry.__init__ (filters by ``available()``)."""

    def test_binary_classification_filters(self, binary_classification_report):
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
        assert "r2" not in names
        assert "rmse" not in names

    def test_regression_filters(self, regression_report):
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

    def test_no_proba_classifier_filters(self, classifier_no_predict_proba_report):
        registry = MetricRegistry(classifier_no_predict_proba_report)
        names = list(registry.keys())
        assert "accuracy" in names
        assert "precision" in names
        assert "recall" in names
        # Without predict_proba/decision_function: roc_auc, log_loss, brier excluded
        assert "roc_auc" not in names
        assert "log_loss" not in names
        assert "brier_score" not in names

    def test_iteration_order_matches_BUILTIN_METRICS(
        self, binary_classification_report
    ):
        registry = MetricRegistry(binary_classification_report)
        # The order is BUILTIN_METRICS' order, restricted to those available
        builtin_order = [
            m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
        ]
        assert list(registry.keys()) == builtin_order


class TestMetricRegistryRepr:
    def test_repr(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        text = repr(registry)
        assert text.startswith("MetricRegistry(")
        assert "accuracy" in text


class TestMetricRegistryAdd:
    """Tests for MetricRegistry.add."""

    def test_add_metric(self, binary_classification_report):
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

    def test_add_position_first(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        m = Metric(name="custom_a", function=None)
        registry.add(m, position="first")
        assert next(iter(registry.keys())) == "custom_a"

    def test_add_position_last(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        m = Metric(name="custom_z", function=None)
        registry.add(m, position="last")
        assert next(reversed(registry.keys())) == "custom_z"

    def test_add_multiple_first_lifo(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        registry.add(Metric(name="a", function=None), position="first")
        registry.add(Metric(name="b", function=None), position="first")
        keys = list(registry.keys())
        assert keys[0] == "b"
        assert keys[1] == "a"

    def test_invalid_position(self, binary_classification_report):
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

    def test_add_builtin_name_conflict(self, binary_classification_report):
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

    def test_add_duplicate_raises(self, binary_classification_report):
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


class TestMetricRegistryRemove:
    """Tests for MetricRegistry.remove."""

    def test_remove_custom(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        registry.add(Metric(name="custom", function=None))
        assert "custom" in registry
        registry.remove("custom")
        assert "custom" not in registry

    def test_remove_builtin(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        assert "accuracy" in registry
        registry.remove("accuracy")
        assert "accuracy" not in registry

    def test_remove_unknown_raises(self, binary_classification_report):
        registry = MetricRegistry(binary_classification_report)
        with pytest.raises(KeyError) as exc_info:
            registry.remove("no_such_metric")
        assert exc_info.value.args[0] == "no_such_metric"

    def test_remove_clears_only_target_cache_entries(
        self, binary_classification_report
    ):
        """Removing a metric clears cache keys whose ``name`` matches and only those."""
        registry = MetricRegistry(binary_classification_report)
        report = binary_classification_report

        registry.add(Metric(name="metric_a", function=None))
        registry.add(Metric(name="metric_b", function=None))

        # Manually populate cache entries
        report._cache[make_cache_key("test", "metric_a", {})] = 0.1
        report._cache[make_cache_key("train", "metric_a", {})] = 0.2
        report._cache[make_cache_key("test", "metric_b", {})] = 0.3

        registry.remove("metric_a")

        assert not any(k[1] == "metric_a" for k in report._cache)
        assert any(k[1] == "metric_b" for k in report._cache)


def test_metric_registry_works_with_test_only_report(
    binary_classification_train_test_split,
):
    """A registry can be built on a report that only has test data."""
    _, X_test, _, y_test = binary_classification_train_test_split
    report = EstimatorReport(
        LogisticRegression().fit(X_test, y_test), X_test=X_test, y_test=y_test
    )
    registry = MetricRegistry(report)
    assert "accuracy" in registry


def test_metric_registry_default_metrics_match_BUILTIN_METRICS(
    binary_classification_report,
):
    """All available built-ins are in the registry; none are missing."""
    registry = MetricRegistry(binary_classification_report)
    available_builtins = {
        m.name for m in BUILTIN_METRICS if m.available(binary_classification_report)
    }
    assert set(registry.keys()) == available_builtins


def test_sklearn_scorer_protocol_recognises_basescorer():
    """``_BaseScorer`` instances satisfy the SKLearnScorer protocol shape."""
    scorer = make_scorer(accuracy_score, response_method="predict")
    assert isinstance(scorer, _BaseScorer)
    assert hasattr(scorer, "_score_func")
    assert hasattr(scorer, "_response_method")
    assert hasattr(scorer, "_kwargs")
