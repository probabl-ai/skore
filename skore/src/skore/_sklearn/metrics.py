from __future__ import annotations

import copy
import inspect
import pickle
from collections import OrderedDict, UserDict
from collections.abc import Callable
from enum import Enum, auto
from inspect import Parameter
from itertools import groupby
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

import numpy as np
import sklearn
import sklearn.metrics
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import _BaseScorer

from skore._sklearn.types import DataSource, PositiveLabel
from skore._utils._cache_key import make_cache_key
from skore._utils._callable import _callable_name

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


class MetricRow(TypedDict):
    """A single row of a metric output.

    Parameters
    ----------
    metric_verbose_name : str
        Human-readable metric name.

    greater_is_better : bool or None
        Whether higher values are better.

    score : float
        Scalar metric value.

    label : label, default=None
        Class label for per-class classification metrics.

    average : str, default=None
        Averaging mode when a metric is aggregated across labels or outputs.

    output : int, default=None
        Output index for multioutput regression metrics.
    """

    metric_verbose_name: str
    greater_is_better: bool | None
    score: float
    label: PositiveLabel | None
    average: str | None
    output: int | None


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

    Notes
    -----
    A metric's value flows through four layers, from raw to human-readable:

    - :meth:`_raw` performs the actual computation.
    - :meth:`_raw_cached` wraps :meth:`_raw` and caches the result.
    - :meth:`rows` outputs metric scores in a structured format.
    - :meth:`pretty` outputs metric scores in a human-readable format, which may
      differ from what the base metric returned.
    """

    kwargs: dict[str, Any] = {}

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
        self.kwargs = kwargs or self.kwargs

        if name is None:
            return

        self.name = name
        self.verbose_name = verbose_name or name.replace("_", " ").title()
        self.greater_is_better = greater_is_better
        self.response_method = response_method
        self.function = function
        self.function_kind = function_kind

    @staticmethod
    def new(
        metric: MetricLike | Metric,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool = True,
        kwargs: dict[str, Any] | None = None,
    ) -> Metric:
        """Create a :class:`Metric` from a metric-like object.

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

    def __repr__(self) -> str:
        """Return a representation of the metric."""
        args = [
            "name",
            "verbose_name",
            "function",
            "greater_is_better",
            "response_method",
            "kwargs",
        ]

        kwargs = [f"{a}={getattr(self, a)!r}" for a in args if hasattr(self, a)]

        return f"Metric({', '.join(kwargs)})"

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-safe representation of the metric."""
        state = self.__dict__.copy()
        if state.get("function") is not None:
            try:
                pickle.dumps(state["function"])
            except Exception:
                # function may be a closure or some other non-picklable object
                state["function"] = None
        return state

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        """Whether this metric is applicable to the given report.

        Override this when the metric is not defined for every ML task or estimator
        (e.g. accuracy is not a regression metric).
        """
        return True

    def _raw(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> Any:
        """Compute the raw metric on the given report.

        Override this when the score cannot be expressed as ``self.function(...)``
        (see :class:`Score`, :class:`FitTime`, :class:`PredictTime`). Subclasses that
        only need to adjust kwargs before delegating should call ``super()._raw(...)``
        (see :class:`Precision`, :class:`Brier`, :class:`RocAuc`).

        Parameters
        ----------
        report : EstimatorReport
            The report to compute the metric for.

        data_source : {"test", "train"}, default="test"
            Which data split to use.

        **kwargs
            Additional keyword arguments passed to the scoring function.

        Returns
        -------
        float or ndarray or dict
            The computed metric value.
        """
        if self.function is None:
            raise ValueError(f"Metric {self.name!r} has no scoring function.")

        metric_params = inspect.signature(self.function).parameters
        call_kwargs = kwargs.copy()
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

            return cast(MetricCallable, self.function)(y_true, y_pred, **call_kwargs)

        assert self.function_kind == FunctionKind.SCORER

        data, y_true = report._get_data_and_y_true(data_source=data_source)
        X = data["_skrub_X"]
        return cast(ScorerCallable, self.function)(
            report.estimator_,
            X,
            y_true,
            **call_kwargs,
        )

    def _raw_cached(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> Any:
        """Compute the raw metric and cache it in the report."""
        cache_key = make_cache_key(data_source, self.name, kwargs)
        score = report._cache.get(cache_key)
        if score is None:
            score = self._raw(report=report, data_source=data_source, **kwargs)

            report._cache[cache_key] = score

        return score

    def _row(
        self,
        *,
        score: Any,
        label: PositiveLabel | None = None,
        average: str | None = None,
        output: int | None = None,
    ) -> MetricRow:
        """Build a single :class:`MetricRow`."""
        return MetricRow(
            metric_verbose_name=self.verbose_name,
            greater_is_better=self.greater_is_better,
            score=score.item() if hasattr(score, "item") else score,
            label=label,
            average=average,
            output=output,
        )

    def _to_rows(
        self,
        score,
        *,
        report: EstimatorReport,
        **kwargs: Any,
    ) -> list[MetricRow]:
        """Convert a score into one or more rows."""
        if isinstance(score, dict):
            # Multimetric scorer
            result = []
            for submetric_name, submetric_value in score.items():
                rows = self._to_rows(submetric_value, report=report, kwargs=kwargs)
                for r in rows:
                    r["metric_verbose_name"] = submetric_name
                result.extend(rows)
            return result

        if (
            report._ml_task == "binary-classification"
            and kwargs.get("average") == "binary"
        ):
            return [
                self._row(score=score, label=kwargs.get("pos_label", report.pos_label))
            ]
        if report._ml_task in ("binary-classification", "multiclass-classification"):
            if isinstance(score, np.ndarray):
                return [
                    self._row(score=s, label=label)
                    for label, s in zip(
                        report.learner_.classes_.tolist(),
                        score.tolist(),
                        strict=False,
                    )
                ]
            return [self._row(score=score, average=kwargs.get("average"))]
        if report._ml_task == "multioutput-regression":
            if isinstance(score, np.ndarray):
                return [
                    self._row(score=s, output=idx)
                    for idx, s in enumerate(score.tolist())
                ]
            return [self._row(score=score, average=kwargs.get("multioutput"))]
        return [self._row(score=score)]

    def rows(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource,
        **kwargs: Any,
    ) -> list[MetricRow]:
        """Compute the metric and expand it into one or more rows.

        Parameters
        ----------
        report : EstimatorReport
            The report to compute the metric for.

        data_source : {"test", "train"}, default="test"
            Which data split to use.

        **kwargs
            Additional keyword arguments passed to the scoring function.

        Returns
        -------
        list of :class:`MetricRow`
            The computed metric value(s).
        """
        merged_kwargs = self.kwargs | kwargs
        score = self._raw_cached(
            report=report, data_source=data_source, **merged_kwargs
        )
        return self._to_rows(score, report=report, **merged_kwargs)

    def _to_pretty(self, rows: list[MetricRow]) -> Any:
        """Convert rows into a human-readable metric output."""
        if len(rows) == 1:
            return rows[0]["score"]

        if len({row["metric_verbose_name"] for row in rows}) != 1:
            # Multi-metric scorer
            # We assume each submetric's values are grouped together
            return {
                name: self._to_pretty(list(rows_))
                for name, rows_ in groupby(
                    rows, key=lambda row: row["metric_verbose_name"]
                )
            }

        if rows[0]["label"] is not None:
            # Multi-class classification
            return {row["label"]: row["score"] for row in rows}

        # Multioutput regression
        # We assume rows are sorted by output
        return np.array([row["score"] for row in rows])

    def pretty(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> Any:
        """Compute the metric in a human-readable shape."""
        rows = self.rows(report=report, data_source=data_source, **kwargs)
        return self._to_pretty(rows)


class FitTime(Metric):
    name = "fit_time"
    verbose_name = "Fit time (s)"
    greater_is_better = False
    function = None
    function_kind = None
    kwargs = {"cast": True}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return True

    def _raw(self, *, report: EstimatorReport, data_source="test", **kwargs):
        if kwargs["cast"] and report._fit_time is None:
            return float("nan")
        return report._fit_time


class PredictTime(Metric):
    name = "predict_time"
    verbose_name = "Predict time (s)"
    greater_is_better = False
    function = None
    function_kind = None
    kwargs = {"cast": True}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return True

    def _raw(self, *, report: EstimatorReport, data_source="test", **kwargs):
        predict_time = report._predict_time.get(data_source)
        if predict_time is None:
            return float("nan") if kwargs["cast"] else None
        return predict_time


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
    kwargs = {"average": None}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")

    def _raw(self, *, report: EstimatorReport, data_source="test", **kwargs):
        if report._ml_task == "binary-classification":
            if kwargs["average"] is None and report.pos_label is not None:
                kwargs["average"] = "binary"
            elif kwargs["average"] != "binary":
                kwargs["pos_label"] = None

        return super()._raw(report=report, data_source=data_source, **kwargs)


class Recall(Metric):
    name = "recall"
    verbose_name = "Recall"
    function = staticmethod(sklearn.metrics.recall_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC
    kwargs = {"average": None}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")

    def _raw(self, *, report: EstimatorReport, data_source="test", **kwargs):
        if report._ml_task == "binary-classification":
            if kwargs["average"] is None and report.pos_label is not None:
                kwargs["average"] = "binary"
            elif kwargs["average"] != "binary":
                kwargs["pos_label"] = None

        return super()._raw(report=report, data_source=data_source, **kwargs)


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
            report.learner_, "predict_proba"
        )

    def _raw(self, *, report: EstimatorReport, data_source="test", **kwargs):
        # The Brier score in scikit-learn requests `pos_label` to ensure that
        # the integral encoding of `y_true` corresponds to the probabilities of
        # the `pos_label`.
        return super()._raw(
            report=report,
            data_source=data_source,
            pos_label=report.learner_.classes_[-1],
            **kwargs,
        )


class RocAuc(Metric):
    name = "roc_auc"
    verbose_name = "ROC AUC"
    response_method = ("predict_proba", "decision_function")
    greater_is_better = True
    function_kind = FunctionKind.METRIC
    kwargs = {"average": None, "multi_class": "ovr"}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        has_predict_proba = hasattr(report.learner_, "predict_proba")
        has_decision_function = hasattr(report.learner_, "decision_function")
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
        ) and hasattr(report.learner_, "predict_proba")


class R2(Metric):
    name = "r2"
    verbose_name = "R²"
    function = staticmethod(sklearn.metrics.r2_score)
    response_method = "predict"
    greater_is_better = True
    function_kind = FunctionKind.METRIC
    kwargs = {"multioutput": "raw_values"}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")


class Rmse(Metric):
    name = "rmse"
    verbose_name = "RMSE"
    function = staticmethod(sklearn.metrics.root_mean_squared_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC
    kwargs = {"multioutput": "raw_values"}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")


class Mae(Metric):
    name = "mae"
    verbose_name = "MAE"
    function = staticmethod(sklearn.metrics.mean_absolute_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC
    kwargs = {"multioutput": "raw_values"}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")


class Mape(Metric):
    name = "mape"
    verbose_name = "MAPE"
    function = staticmethod(sklearn.metrics.mean_absolute_percentage_error)
    response_method = "predict"
    greater_is_better = False
    function_kind = FunctionKind.METRIC
    kwargs = {"multioutput": "raw_values"}

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("regression", "multioutput-regression")


class Score(Metric):
    name = "score"
    verbose_name = "Score"
    greater_is_better = True
    function = None
    function_kind = None

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return hasattr(report.estimator_, "score")

    def _raw(
        self,
        *,
        report: EstimatorReport,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> Any:
        # Both estimator paths accept the dict ``data`` directly:
        # ``_LearnerAdapter`` unpacks ``_skrub_X``/``_skrub_y`` for sklearn
        # estimators; ``SkrubLearner`` takes the full env, preserving vars
        # beyond X/y (e.g. additional tables referenced by the DataOp).
        data, _ = report._get_data_and_y_true(data_source=data_source)
        return report.learner_.score(data, **kwargs)


# Order matters for default display
BUILTIN_METRICS: list[Metric] = [
    Score(),
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

        # Needs to be called ``data`` since we inherit from :class:`UserDict`
        self.data = OrderedDict(
            (metric.name, metric)
            for metric in BUILTIN_METRICS
            if metric.available(report)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.data.keys())})"

    def __missing__(self, key: str) -> Metric:
        stripped = key.removeprefix("neg_")
        if stripped != key and stripped in self.data:
            return self.data[stripped]
        raise KeyError(f"{key!r} not found in the registered metrics")

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

    def remove(self, *, report: EstimatorReport, name: str) -> None:
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
        del self.data[name]

        keys_to_delete = [k for k in report._cache if k[1] == name]
        for k in keys_to_delete:
            del report._cache[k]
