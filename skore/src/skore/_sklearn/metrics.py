from __future__ import annotations

import copy
import inspect
import warnings
from collections import UserDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import sklearn
import sklearn.metrics
from sklearn.metrics._scorer import _BaseScorer

from skore._sklearn.types import DataSource, MetricLike, PositiveLabel
from skore._utils._cache_key import make_cache_key
from skore._utils._callable_name import _callable_name

if TYPE_CHECKING:
    from skore import EstimatorReport


def _select_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter `kwargs` based on the kwargs that `func` accepts."""
    return {
        param: kwargs[param]
        for param in inspect.signature(func).parameters
        if param in kwargs
    }


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


class Metric:
    """A metric that can compute a score from a report.

    Subclass this to define built-in metrics with class-level attributes.
    For custom (user-registered) metrics, instantiate directly.

    Parameters
    ----------
    name : str
        Technical name used for lookup (e.g. ``"accuracy"``).

    verbose_name : str
        Display name shown in reports (e.g. ``"Accuracy"``).

    greater_is_better : bool or None
        Whether a higher value is better.

    score_func : callable or None
        The scoring function ``(y_true, y_pred, **kw) -> float``.

    response_method : str, list of str, or None, default="predict"
        Estimator method to get predictions.

    kwargs : dict, default={}
        Default keyword arguments for ``score_func``.

    source_code : str or None, default=None
        Source code of the score function, captured at registration time.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool | None = None,
        score_func: Callable | None = None,
        response_method: str | list[str] | tuple[str, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        source_code: str | None = None,
    ):
        # When name is None, the metric is being instantiated from a subclass
        # (e.g. Accuracy()) whose fields are defined as class attributes.
        # Only `kwargs` needs to be set as an instance attribute.
        self.kwargs = kwargs or {}

        if name is None:
            return

        self.name = name
        self.verbose_name = verbose_name or name.replace("_", " ").title()
        self.score_func = score_func
        self.response_method: str | list[str] | tuple[str, ...] | None = (
            response_method or "predict"
        )
        self.greater_is_better = greater_is_better
        self.source_code = source_code

    @property
    def icon(self) -> str:
        """Favorability icon derived from ``greater_is_better``."""
        match self.greater_is_better:
            case True:
                return "(↗︎)"
            case False:
                return "(↘︎)"
            case _:
                return ""

    def is_callable(self) -> bool:
        """Return whether the score function is callable."""
        return self.score_func is not None and callable(self.score_func)

    def __copy__(self):
        """Shallow copy that preserves all attributes (including score_func)."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("score_func") is not None:
            import pickle as _pickle

            try:
                _pickle.dumps(state["score_func"])
            except Exception:
                state["score_func"] = None
                state["_score_func_lost"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_score_func_lost", False):
            warnings.warn(
                f"The score function for metric {state.get('name', '?')!r} "
                "could not be restored after pickling (e.g., it was a lambda "
                "or closure).",
                UserWarning,
                stacklevel=2,
            )
        self.__dict__.update(state)

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        """Whether this metric is applicable to the given report."""
        return True

    def _prediction_pos_label(self, report: EstimatorReport) -> PositiveLabel | None:
        """Return the pos_label to use for ``_get_predictions``.

        Subclasses may override this to provide a default when ``report.pos_label``
        is ``None`` (e.g. binary classification metrics that need 1-D predictions).
        """
        return report.pos_label

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"verbose_name={self.verbose_name!r}"
            ")"
        )

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
            Additional keyword arguments passed to ``score_func``.
        """
        if self.score_func is None:
            raise NotImplementedError(
                f"Metric {self.name!r} has no score_func and must override __call__."
            )

        _, y_true = report._get_data_and_y_true(data_source=data_source)

        # Merge default kwargs with call-time kwargs
        merged_kwargs = self.kwargs | kwargs

        cache_key = make_cache_key(data_source, self.name, merged_kwargs)
        score = report._cache.get(cache_key)
        if score is not None:
            return score

        assert self.response_method is not None

        y_pred = report._get_predictions(
            data_source=data_source,
            response_method=self.response_method,
            pos_label=self._prediction_pos_label(report),
        )

        metric_params = inspect.signature(self.score_func).parameters
        call_kwargs = merged_kwargs.copy()
        if "pos_label" in metric_params and "pos_label" not in call_kwargs:
            call_kwargs["pos_label"] = report.pos_label

        score = self.score_func(y_true, y_pred, **call_kwargs)

        if isinstance(score, np.ndarray):
            score = score.tolist()

        if hasattr(score, "item"):
            score = score.item()
        elif isinstance(score, list):
            if len(score) == 1:
                score = score[0]
            elif "classification" in report._ml_task:
                score = dict(
                    zip(report._estimator.classes_.tolist(), score, strict=False)
                )

        report._cache[cache_key] = score
        return score


class FitTime(Metric):
    name = "fit_time"
    verbose_name = "Fit time (s)"
    greater_is_better = False
    response_method = None
    score_func = None

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
    response_method = None
    score_func = None

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
    score_func = staticmethod(sklearn.metrics.accuracy_score)
    response_method = "predict"
    greater_is_better = True

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in ("binary-classification", "multiclass-classification")


class Precision(Metric):
    name = "precision"
    verbose_name = "Precision"
    score_func = staticmethod(sklearn.metrics.precision_score)
    response_method = "predict"
    greater_is_better = True

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
    score_func = staticmethod(sklearn.metrics.recall_score)
    response_method = "predict"
    greater_is_better = True

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
    score_func = staticmethod(sklearn.metrics.brier_score_loss)
    response_method = "predict_proba"
    greater_is_better = False

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task == "binary-classification" and hasattr(
            report._estimator, "predict_proba"
        )

    def _prediction_pos_label(self, report):
        return report._estimator.classes_[-1]

    def __call__(self, *, report: EstimatorReport, data_source="test", **kwargs):
        # The Brier score in scikit-learn requests `pos_label` to ensure that
        # the integral encoding of `y_true` corresponds to the probabilities of
        # the `pos_label`. We pass the same `pos_label` to `_get_predictions`
        # (via `_prediction_pos_label`) and to the metric itself.
        return super().__call__(
            report=report,
            data_source=data_source,
            pos_label=self._prediction_pos_label(report),
            **kwargs,
        )


class RocAuc(Metric):
    name = "roc_auc"
    verbose_name = "ROC AUC"
    score_func = staticmethod(sklearn.metrics.roc_auc_score)
    response_method = ("predict_proba", "decision_function")
    greater_is_better = True

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        has_predict_proba = hasattr(report._estimator, "predict_proba")
        has_decision_function = hasattr(report._estimator, "decision_function")
        if report._ml_task == "binary-classification":
            return has_predict_proba or has_decision_function
        elif report._ml_task == "multiclass-classification":
            return has_predict_proba
        return False

    def _prediction_pos_label(self, report):
        if report._ml_task == "multiclass-classification":
            return None
        if report.pos_label is None:
            return report._estimator.classes_[-1]
        return report.pos_label

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
    score_func = staticmethod(sklearn.metrics.log_loss)
    response_method = "predict_proba"
    greater_is_better = False

    @staticmethod
    def available(report: EstimatorReport) -> bool:
        return report._ml_task in (
            "binary-classification",
            "multiclass-classification",
        ) and hasattr(report._estimator, "predict_proba")


class R2(Metric):
    name = "r2"
    verbose_name = "R²"
    score_func = staticmethod(sklearn.metrics.r2_score)
    response_method = "predict"
    greater_is_better = True

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
    score_func = staticmethod(sklearn.metrics.root_mean_squared_error)
    response_method = "predict"
    greater_is_better = False

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
    FitTime(),
    PredictTime(),
]


class MetricRegistry(UserDict):
    """Registry of metric instances for a report.

    Parameters
    ----------
    report : EstimatorReport
        The parent report.
    """

    def __init__(self, report: EstimatorReport):
        """Construct a MetricRegistry.

        The report is analyzed to filter metrics depending on the report's
        characteristics (e.g. the ML task and the estimator's prediction methods).
        """
        super().__init__()
        self._report = report

        # Needs to be called `data` since we inherit from UserDict
        self.data = {
            metric.name: metric
            for metric in BUILTIN_METRICS
            if metric.available(report)
        }

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.data.keys())})"

    def check_metric(self, metric: MetricLike, metric_kwargs: dict[str, Any]) -> Metric:
        """Convert a single "metric-like" to a Metric.

        Parameters
        ----------
        metric : string or SKLearnScorer or callable (y_true, y_pred) -> score
            The metric to parse.
        metric_kwargs : dict
            Kwargs to pass; each metric takes only the kwargs it accepts.
        """
        if isinstance(metric, _BaseScorer):
            func_name = metric._score_func.__name__

            kwargs = metric._kwargs.copy()
            if "pos_label" in inspect.signature(metric._score_func).parameters:
                if (
                    "pos_label" in kwargs
                    and self._report.pos_label != kwargs["pos_label"]
                ):
                    raise ValueError(
                        "The `pos_label` passed in the scorer "
                        "and the one used when creating the report must match; "
                        f"got {kwargs['pos_label']!r} and {self._report.pos_label!r}."
                    )
                kwargs["pos_label"] = self._report.pos_label

            return Metric(
                name=func_name,
                greater_is_better=metric._sign == 1,
                score_func=metric._score_func,
                response_method=metric._response_method,
                kwargs=kwargs,
            )
        elif metric in self:
            parsed_metric = copy.copy(self[metric])
            parsed_metric.kwargs = (
                _select_kwargs(parsed_metric.score_func, metric_kwargs)
                if parsed_metric.score_func is not None
                else {}
            )
            return parsed_metric
        elif isinstance(metric, str):
            if len(metric_kwargs) != 0:
                raise ValueError(
                    "The `metric_kwargs` parameter is not supported when "
                    "`metric` is a scikit-learn scorer name. Use the function "
                    "`sklearn.metrics.make_scorer` to create a scorer with "
                    "additional parameters."
                )

            metric = _METRIC_ALIASES.get(metric, metric)

            try:
                scorer = sklearn.metrics.get_scorer(metric)
            except ValueError as err:
                raise ValueError(
                    f"Invalid metric: {metric!r}. "
                    "Please use a valid metric from the list of supported "
                    f"metrics: {list(self.keys())} "
                    "or a valid scikit-learn metric string: "
                    f"{sklearn.metrics.get_scorer_names()}."
                ) from err

            return self.check_metric(scorer, metric_kwargs)
        elif callable(metric):
            if "response_method" not in metric_kwargs:
                raise ValueError(
                    "response_method is required when the metric is a "
                    "callable. Pass it directly or through `metric_kwargs`."
                )

            return Metric(
                name=metric.__name__,
                greater_is_better=metric_kwargs.get("greater_is_better"),
                score_func=metric,
                response_method=metric_kwargs["response_method"],
                kwargs=_select_kwargs(metric, metric_kwargs),
            )
        else:
            raise ValueError(f"Invalid type of metric: {type(metric)} for {metric!r}")
