from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, runtime_checkable, Protocol

from numpy.typing import ArrayLike
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import accuracy_score, precision_score, r2_score, get_scorer

from skore import EstimatorReport
from skore._sklearn._base import _BaseAccessor, _get_cached_response_values
from skore._sklearn.types import DataSource, PositiveLabel


@runtime_checkable
class MetricProtocol(Protocol):
    name: str
    verbose_name: str
    score_func: Callable  # foo(y_true, y_pred) -> float | any
    response_method: str | list[str] | tuple[str, ...]
    greater_is_better: bool
    custom: bool


class Metric(_BaseAccessor[EstimatorReport], MetricProtocol):  # or Protocol
    def __init__(self, report, /):
        super().__init__(report)
        self.report = report

    def _get_help_tree_title(self) -> str:
        pass

    @staticmethod
    def available(report) -> bool:
        return True

    def __call__(
        self,
        *,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        pos_label: PositiveLabel | None = None,
        **kwargs: Any,
    ) -> float | list[float] | dict[Any, float]:
        if data_source_hash is None:
            X, y, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y,
            )

        results = _get_cached_response_values(
            cache=self.report._cache,
            estimator_hash=int(self.report._hash),
            estimator=self.report.estimator_,
            X=X,
            response_method=self.response_method,
            pos_label=pos_label,
            data_source=data_source,
            data_source_hash=data_source_hash,
        )

        for key_tuple, value, is_cached in results:
            if not is_cached:
                self.report._cache[key_tuple] = value

            if key_tuple[-1] != "predict_time":
                y_pred = value

        sign = -1 if self.name.startswith("neg_") else 1
        score = sign * self.score_func(y, y_pred, **kwargs)

        if hasattr(score, "tolist"):
            score = score.tolist()
        elif hasattr(score, "item"):
            score = score.item()

        if isinstance(score, list):
            if "classification" in self.report.ml_task:
                classes = self.report._estimator.classes_.tolist()
                return dict(zip(classes, score, strict=False))

            if len(score) == 1:
                return score[0]

        return score

    @staticmethod
    def factory(
        report, /, *, name, verbose_name, score_func, response_method, greater_is_better
    ) -> Metric:
        metric = Metric(report)

        metric.name = name
        metric.verbose_name = verbose_name
        metric.score_func = score_func
        metric.response_method = response_method
        metric.greater_is_better = greater_is_better
        metric.custom = True

        return metric


class Accuracy(Metric):
    name = "accuracy"
    verbose_name = "Accuracy"
    score_func = staticmethod(accuracy_score)
    response_method = "predict"
    greater_is_better = True
    custom = False

    @staticmethod
    def available(report) -> bool:
        return report.ml_task in ("binary-classification", "multiclass-classification")


Average = Literal["binary", "macro", "micro", "weighted", "samples"]


class Precision(Metric):
    name = "precision"
    verbose_name = "Precision"
    score_func = staticmethod(precision_score)
    response_method = "predict"
    greater_is_better = True
    custom = False

    @staticmethod
    def available(report) -> bool:
        return report.ml_task in ("binary-classification", "multiclass-classification")

    def __call__(
        self,
        *,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        data_source: DataSource = "test",
        pos_label: PositiveLabel | None | ... = ...,
        average: Average | None = None,
    ) -> float | dict[Any, float]:
        if pos_label is ...:
            pos_label = self.report.pos_label

        if (
            (average is None)
            and (self.report.ml_task == "binary-classification")
            and (pos_label is not None)
        ):
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        return super().__call__(
            X=X,
            y=y,
            data_source=data_source,
            pos_label=pos_label,
            average=average,
        )


class R2(Metric):
    name = "r2"
    verbose_name = "r²"
    score_func = staticmethod(r2_score)
    response_method = "predict"
    greater_is_better = True
    custom = False

    @staticmethod
    def available(report) -> bool:
        return report.ml_task in ("regression", "multioutput-regression")

    def __call__(
        self,
        *,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        data_source: DataSource = "test",
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list[float]:
        return super().__call__(
            X=X,
            y=y,
            data_source=data_source,
            multioutput=multioutput,
        )


class MetricRegistry:
    def __init__(self, report, /, *metric_classes):
        super().__init__()

        self.__report = report
        self.__metric_name_to_function = {}

        for metric_class in metric_classes:
            if not metric_class.available(self.__report):
                continue

            metric = metric_class(self.__report)
            self.__metric_name_to_function[metric.name] = metric

    def __iter__(self):
        yield from self.__metrics_name_to_function.items()

    def __repr__(self):
        return str(self.__metric_name_to_function)

    def __getattr__(self, name):
        return self.__metric_name_to_function[name]

    def summary(self):
        raise NotImplementedError

    def append(
        self,
        metric: str | _BaseScorer | Callable,
        /,
        *,
        response_method="predict",
        greater_is_better=True,
    ):
        if isinstance(metric, str | _BaseScorer):
            if isinstance(metric, str):
                metric = get_scorer(metric)

            metric = Metric.factory(
                self.__report,
                name=metric._score_func.__name__,
                verbose_name=metric._score_func.__name__.replace("_", " ").title(),
                score_func=partial(metric._score_func, **metric._kwargs),
                response_method=metric._response_method,
                greater_is_better=bool(metric._sign),
            )
        elif callable(metric):
            metric = Metric.factory(
                self.__report,
                name=metric.__name__,
                verbose_name=metric.__name__.replace("_", " ").title(),
                score_func=metric,
                response_method=response_method,
                greater_is_better=greater_is_better,
            )
        else:
            raise Exception

        self.__metric_name_to_function[metric.name] = metric


from skore import EstimatorReport


class NewEstimatorReport(EstimatorReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # create registry and append default metrics
        self.metrics = MetricRegistry(self, Accuracy, Precision, R2)
