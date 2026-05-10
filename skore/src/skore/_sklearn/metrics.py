from __future__ import annotations

import copy
import inspect
import pickle
from collections import OrderedDict, UserDict
from collections.abc import Callable
from enum import Enum, auto
from inspect import Parameter
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import sklearn
import sklearn.metrics
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import _BaseScorer

from skore._sklearn.types import DataSource, PositiveLabel
from skore._utils._cache_key import make_cache_key
from skore._utils._callable_name import _callable_name

if TYPE_CHECKING:
    from skore import EstimatorReport


class SKLearnScorer(Protocol):
    """Protocol defining the interface of scikit-learn's _BaseScorer."""

    _score_func: Callable
    _response_method: str | list[str]
    _kwargs: dict[str, Any]


MetricReturnValue = float | ArrayLike
MetricCallable = Callable[[ArrayLike, ArrayLike], MetricReturnValue]
ScorerCallable = Callable[[BaseEstimator, ArrayLike, ArrayLike], MetricReturnValue]
MetricLike = str | ScorerCallable | SKLearnScorer

_METRIC_ALIASES: dict[str, str] = {
    "mean_squared_error": "neg_mean_squared_error",
    "mean_absolute_error": "neg_mean_absolute_error",
    "mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
    "median_absolute_error": "neg_median_absolute_error",
    "mean_squared_log_error": "neg_mean_squared_log_error",
    "root_mean_squared_error": "neg_root_mean_squared_error",
    "root_mean_squared_log_error": "neg_root_mean_squared_log_error",
    "mean_poisson_deviance": "neg_mean_poisson_deviance",
    "mean_gamma_deviance": "neg_mean_gamma_deviance",
    "max_error": "neg_max_error",
    "negative_likelihood_ratio": "neg_negative_likelihood_ratio",
}


class FunctionKind(Enum):
    """Kind of scoring function, in the sklearn sense.

    A metric is a callable of the form ``(y_true, y_pred, **kwargs) -> score``.
    A scorer is a callable of the form ``(estimator, X, y_true, **kwargs) -> score``.
    """

    METRIC = auto()
    SCORER = auto()


class MissingKwargsError(Exception):
    def __init__(self, metric, missing_kwargs):
        self.metric = _callable_name(metric)
        self.missing_kwargs = missing_kwargs
        self.msg = (
            f"Callable {self.metric!r} has required "
            f"parameter(s) {tuple(self.missing_kwargs)} not covered by the "
            f"provided kwargs."
        )

    def __str__(self):
        return self.msg


class Metric:
    """A metric that can compute a score from a report.

    Parameters
    ----------
    name : str
        Technical name used for lookup (e.g. ``"accuracy"``).

    verbose_name : str
        Display name shown in reports (e.g. ``"Accuracy"``).

    greater_is_better : bool or None
        Whether a higher value is better.

    response_method : str, list of str, or None, default="predict"
        Estimator method to get predictions.

    function : callable or None
        Scoring function.

    function_kind : FunctionKind or None, default=None
        Kind of scoring function (either metric or scorer).

    kwargs : dict, default={}
        Default keyword arguments for the scoring function.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool | None = None,
        response_method: str | list[str] | tuple[str, ...] | None = None,
        function: ScorerCallable | MetricCallable | None = None,
        function_kind: FunctionKind | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        """Construct a Metric.

        Not meant to be executed directly; instead use `Metric.new` to instantiate a
        new Metric.
        """
        # When name is None, the metric is being instantiated from a subclass
        # (e.g. Accuracy()) whose fields are defined as class attributes.
        # Only `kwargs` needs to be set as an instance attribute.
        self.kwargs = kwargs or {}

        if name is None:
            return

        self.name = name
        self.verbose_name = verbose_name or name.replace("_", " ").title()
        self.greater_is_better = greater_is_better
        self.response_method = response_method
        self.function = function
        self.function_kind = function_kind

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("function") is not None:
            try:
                pickle.dumps(state["function"])
            except Exception:
                state["function"] = None
        return state

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        """Whether this metric is applicable to the given report."""
        return True

    def __repr__(self) -> str:
        args = [
            f"name={self.name!r}",
            f"verbose_name={self.verbose_name!r}",
            f"function={self.function}",
            f"greater_is_better={self.greater_is_better}",
            f"response_method={self.response_method}",
            f"kwargs={self.kwargs}",
        ]

        return f"Metric({', '.join(args)})"

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> float | dict[PositiveLabel, float] | list:
        """Compute the metric score.

        Parameters
        ----------
        report : EstimatorReport
            The report to compute the metric for.

        data_source : {"test", "train"}, default="test"
            Which data split to use.

        **kwargs
            Additional keyword arguments passed to the scoring function.
        """
        # Merge default kwargs with call-time kwargs
        merged_kwargs = self.kwargs | kwargs

        cache_key = make_cache_key(data_source, self.name, merged_kwargs)
        score = report._cache.get(cache_key)
        if score is not None:
            return score

        if self.function is None:
            raise ValueError(f"Metric {self.name!r} has no scoring function.")

        metric_params = inspect.signature(self.function).parameters
        call_kwargs = merged_kwargs.copy()
        if "pos_label" in metric_params and "pos_label" not in call_kwargs:
            call_kwargs["pos_label"] = report.pos_label

        if self.function_kind == FunctionKind.METRIC:
            assert self.response_method is not None

            _, y_true = report._get_data_and_y_true(data_source=data_source)
            y_pred = report._get_predictions(
                data_source=data_source,
                response_method=self.response_method,
                pos_label=call_kwargs.get("pos_label", None),
            )

            score = cast(MetricCallable, self.function)(y_true, y_pred, **call_kwargs)
        elif self.function_kind == FunctionKind.SCORER:
            data, y_true = report._get_data_and_y_true(data_source=data_source)
            X = data["_skrub_X"]

            score = cast(ScorerCallable, self.function)(
                report.estimator_,
                X,
                y_true,
                **call_kwargs,
            )

        if isinstance(score, np.ndarray):
            score = cast(NDArray, score).tolist()

        if hasattr(score, "item"):
            score = cast(NDArray, score).item()
        elif isinstance(score, list):
            if len(score) == 1:
                score = score[0]
            elif "classification" in report._ml_task:
                score = dict(
                    zip(report._estimator.classes_.tolist(), score, strict=False)
                )

        report._cache[cache_key] = score
        return cast(float | dict[PositiveLabel, float] | list[float], score)

    @staticmethod
    def new(
        metric: MetricLike | Metric,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> Metric:
        """Convert a metric-like object into a :class:`Metric` instance.

        Parameters
        ----------
        metric : str, callable, sklearn scorer, or Metric
            The metric to convert.

            - If a string, will be converted via
              :func:`sklearn.metrics.get_scorer`.
              Scikit-learn metrics that require a ``neg_`` prefix (e.g.
              ``"neg_mean_squared_error"``) can also be passed without it
              (e.g. ``"mean_squared_error"``); the alias is resolved
              automatically.
            - If a callable, expected to be of the form
              ``(estimator, X, y, **kw) -> float``.
            - If a sklearn scorer, expected to be a _BaseScorer instance
              (e.g. as returned by :func:`sklearn.metrics.make_scorer` and
              :func:`sklearn.metrics.get_scorer`).
            - If a `Metric`, will return a copy.

        name : str, optional
            Custom name for the metric. If not provided the name is inferred
            from the input (e.g. the function's ``__name__``).

        verbose_name : str, optional
            Custom verbose name for the metric which will be used for display purposes.
            If not provided, will be inferred from the metric name.

        greater_is_better : bool, default=True
            Whether a higher score is better. Only used when *metric* is a
            scorer callable.

        kwargs : dict, optional
            Default keyword arguments passed to the score function at call
            time. Only used when *metric* is a plain callable. For sklearn
            scorers, kwargs are extracted from the scorer itself.

        Returns
        -------
        Metric
            A new :class:`Metric` instance.
        """
        if isinstance(metric, Metric):
            if name is None and verbose_name is None:
                return metric

            result = copy.copy(metric)

            if name is not None:
                result.name = name
                result.verbose_name = name.replace("_", " ").title()

            if verbose_name is not None:
                result.verbose_name = verbose_name

            return result

        elif isinstance(metric, _BaseScorer):
            return Metric(
                name=name or _callable_name(metric._score_func),
                verbose_name=verbose_name,
                greater_is_better=metric._sign == 1,
                function=metric._score_func,
                response_method=metric._response_method,
                kwargs=metric._kwargs.copy(),
                function_kind=FunctionKind.METRIC,
            )
        elif isinstance(metric, str):
            metric_with_neg = _METRIC_ALIASES.get(metric, metric)

            try:
                scorer = sklearn.metrics.get_scorer(metric_with_neg)
            except ValueError:
                raise ValueError(
                    f"Invalid metric: {metric!r}. "
                    "Please use a valid scikit-learn metric string: "
                    f"{sklearn.metrics.get_scorer_names()}."
                ) from None
            name = name if name is not None else metric.removeprefix("neg_")
            return Metric.new(scorer, name=name)
        elif callable(metric):
            # Fail fast if metric is (y_true, y_pred) -> score
            params = list(inspect.signature(metric).parameters.values())
            if params[0].name.startswith("y"):
                raise TypeError(
                    "Expected a scorer callable with an estimator as its first "
                    f"argument; got first argument {params[0].name!r}"
                )

            positional_args = [
                p.name for p in params if p.kind <= Parameter.POSITIONAL_OR_KEYWORD
            ]
            if len(positional_args) <= 2:
                raise TypeError(
                    "Expected a scorer callable with at least 3 positional "
                    f"arguments (estimator, X, y); got {positional_args}"
                )

            # (estimator, X, y) -> score
            resolved_kwargs = kwargs or {}
            missing_kwargs = [
                param.name
                for param in params[3:]  # estimator, X, y
                if param.default is inspect.Parameter.empty
                and param.name not in resolved_kwargs
            ]
            if missing_kwargs:
                raise MissingKwargsError(metric, missing_kwargs)
            return Metric(
                name=name or _callable_name(metric),
                verbose_name=verbose_name,
                greater_is_better=greater_is_better,
                function=metric,
                kwargs=resolved_kwargs,
                function_kind=FunctionKind.SCORER,
            )
        else:
            raise TypeError(
                f"Cannot create a Metric from {type(metric)!r}. "
                "Expected a callable, sklearn scorer, or Metric instance."
            )


class FitTime(Metric):
    name = "fit_time"
    verbose_name = "Fit time (s)"
    greater_is_better = False
    function = None
    function_kind = None

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return True

    def __call__(
        self, *, report: EstimatorReport, data_source="test", cast=True, **kwargs
    ):
        if cast and report.fit_time_ is None:
            return float("nan")
        return report.fit_time_


class PredictTime(Metric):
    name = "predict_time"
    verbose_name = "Predict time (s)"
    greater_is_better = False
    function = None
    function_kind = None

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return True

    def __call__(
        self, *, report: EstimatorReport, data_source="test", cast=True, **kwargs
    ):
        predict_time_cache_key = make_cache_key(data_source, "predict_time")
        return report._cache.get(
            predict_time_cache_key, (float("nan") if cast else None)
        )


class Accuracy(Metric):
    name = "accuracy"
    verbose_name = "Accuracy"
    function = staticmethod(sklearn.metrics.accuracy_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")


class Precision(Metric):
    name = "precision"
    verbose_name = "Precision"
    function = staticmethod(sklearn.metrics.precision_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")

    def __call__(
        self, *, report: EstimatorReport, data_source="test", average=None, **kwargs
    ):
        if report._ml_task == "binary-classification":
            if average is None and report.pos_label is not None:
                average = "binary"
            elif average != "binary":
                kwargs["pos_label"] = None

        return super().__call__(
            report=report, data_source=data_source, average=average, **kwargs
        )


class Recall(Metric):
    name = "recall"
    verbose_name = "Recall"
    function = staticmethod(sklearn.metrics.recall_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")

    def __call__(
        self, *, report: EstimatorReport, data_source="test", average=None, **kwargs
    ):
        if report._ml_task == "binary-classification":
            if average is None and report.pos_label is not None:
                average = "binary"
            elif average != "binary":
                kwargs["pos_label"] = None

        return super().__call__(
            report=report, data_source=data_source, average=average, **kwargs
        )


class Brier(Metric):
    name = "brier_score"
    verbose_name = "Brier score"
    function = staticmethod(sklearn.metrics.brier_score_loss)
    response_method = "predict_proba"
    greater_is_better = False
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task == "binary-classification" and hasattr(
            report._estimator, "predict_proba"
        )

    def __call__(self, *, report: EstimatorReport, data_source="test", **kwargs):
        # The Brier score in scikit-learn requests `pos_label` to ensure that
        # the integral encoding of `y_true` corresponds to the probabilities of
        # the `pos_label`.
        return super().__call__(
            report=report,
            data_source=data_source,
            pos_label=report._estimator.classes_[-1],
            **kwargs,
        )


class RocAuc(Metric):
    name = "roc_auc"
    verbose_name = "ROC AUC"
    response_method = ("predict_proba", "decision_function")
    greater_is_better = True
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        has_predict_proba = hasattr(report._estimator, "predict_proba")
        has_decision_function = hasattr(report._estimator, "decision_function")
        if report._ml_task == "binary-classification":
            return has_predict_proba or has_decision_function
        elif report._ml_task == "multiclass-classification":
            return has_predict_proba
        return False

    @staticmethod
    def function(y_true, y_score, **kwargs):
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        return sklearn.metrics.roc_auc_score(y_true, y_score, **kwargs)

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source="test",
        average=None,
        multi_class="ovr",
        **kwargs,
    ):
        return super().__call__(
            report=report,
            data_source=data_source,
            average=average,
            multi_class=multi_class,
            **kwargs,
        )


class LogLoss(Metric):
    name = "log_loss"
    verbose_name = "Log loss"
    function = staticmethod(sklearn.metrics.log_loss)
    response_method = "predict_proba"
    greater_is_better = False
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in (
            "binary-classification",
            "multiclass-classification",
        ) and hasattr(report._estimator, "predict_proba")


class R2(Metric):
    name = "r2"
    verbose_name = "R²"
    function = staticmethod(sklearn.metrics.r2_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source="test",
        multioutput="raw_values",
        **kwargs,
    ):
        return super().__call__(
            report=report, data_source=data_source, multioutput=multioutput, **kwargs
        )


class Rmse(Metric):
    name = "rmse"
    verbose_name = "RMSE"
    function = staticmethod(sklearn.metrics.root_mean_squared_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source="test",
        multioutput="raw_values",
        **kwargs,
    ):
        return super().__call__(
            report=report, data_source=data_source, multioutput=multioutput, **kwargs
        )


class Mae(Metric):
    name = "mae"
    verbose_name = "MAE"
    function = staticmethod(sklearn.metrics.mean_absolute_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source="test",
        multioutput="raw_values",
        **kwargs,
    ):
        return super().__call__(
            report=report, data_source=data_source, multioutput=multioutput, **kwargs
        )


class Mape(Metric):
    name = "mape"
    verbose_name = "MAPE"
    function = staticmethod(sklearn.metrics.mean_absolute_percentage_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")

    def __call__(
        self,
        *,
        report: EstimatorReport,
        data_source="test",
        multioutput="raw_values",
        **kwargs,
    ):
        return super().__call__(
            report=report, data_source=data_source, multioutput=multioutput, **kwargs
        )


# Order matters for default display
BUILTIN_METRICS: list[Metric] = [
    Accuracy(),
    Precision(),
    Recall(),
    RocAuc(),
    LogLoss(),
    Brier(),
    R2(),
    Rmse(),
    Mae(),
    Mape(),
    FitTime(),
    PredictTime(),
]


class MetricRegistry(UserDict[str, Metric]):
    """Registry of metric instances for a report.

    Parameters
    ----------
    report : EstimatorReport
        The parent report.
    """

    data: OrderedDict[str, Metric]

    def __init__(self, report: EstimatorReport) -> None:
        """Construct a MetricRegistry.

        The report is analyzed to filter metrics depending on the report's
        characteristics (e.g. the ML task and the estimator's prediction methods).
        """
        super().__init__()
        self._report = report

        # Needs to be called ``data`` since we inherit from :class:`UserDict`.
        self.data = OrderedDict(
            (metric.name, metric)
            for metric in BUILTIN_METRICS
            if metric.available(report)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.data.keys())})"

    def add(
        self,
        metric: Metric,
        *,
        position: Literal["first", "last"] = "first",
    ) -> None:
        """Add a custom metric to the registry.

        Parameters
        ----------
        metric : Metric
            The metric instance to add.

        position : {"first", "last"}, default="first"
            Where to place the metric in iteration order (e.g. default
            :meth:`~skore.EstimatorReport.metrics.summarize` row order).
            ``"first"`` inserts at the front; ``"last"`` at the end.
        """
        if position not in ("first", "last"):
            raise ValueError(f"position must be 'first' or 'last', got {position!r}.")

        if metric.name in {m.name for m in BUILTIN_METRICS}:
            raise ValueError(
                f"Cannot add {metric.name!r}: it is a built-in metric name."
            )

        if metric.name in self.data:
            raise ValueError(
                f"Cannot add {metric.name!r}: it already exists. "
                "Remove it first using the `remove` method."
            )

        self.data[metric.name] = metric

        if position == "first":
            self.data.move_to_end(metric.name, last=False)

    def remove(self, name: str) -> None:
        """Remove a metric from the registry.

        Built-in metrics may be removed; they stay absent for the lifetime of this
        registry (the same instance is kept for the parent report).

        Parameters
        ----------
        name : str
            The technical name of the metric to remove.

        Raises
        ------
        KeyError
            If `name` is not registered.
        """
        if name not in self.data:
            raise KeyError(name)

        keys_to_delete = [k for k in self._report._cache if k[1] == name]
        for k in keys_to_delete:
            del self._report._cache[k]

        del self.data[name]
