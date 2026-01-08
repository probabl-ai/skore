from __future__ import annotations

from abc import ABC
from collections import UserList
from collections.abc import Callable
from typing import Any, Literal

from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, precision_score, r2_score

from skore._sklearn._base import _get_cached_response_values
from skore._sklearn.types import DataSource, PositiveLabel

# https://github.com/scikit-learn/scikit-learn/blob/e9752287ffffecb2d2878ce1ed97a77a43941579/sklearn/metrics/_scorer.py#L228
#
# def __init__(self, score_func, sign, kwargs# , response_method="predict"):
#     self._score_func = score_func
#     self._sign = sign
#     self._kwargs = kwargs
#     self._response_method = response_method
#


class Metric(ABC):  # or Protocol
    # est-ce qu'on hérite de basescorer ? qu'on puisse intergenger un Metric/BaseScorer
    # ou est-ce qu'on fait un wrapper qui a un attribute scorer ?

    NAME: str
    VERBOSE_NAME: str
    SCORE_FUNC: Callable  # foo(y_true, y_pred) -> float | Any
    RESPONSE_METHOD: str | list[str] | tuple[str, ...]
    GREATER_IS_BETTER: bool
    CUSTOM: bool

    def __init__(self, report, /):
        self.report = report

    @staticmethod
    def available(report) -> bool:
        raise NotImplementedError

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
        # changer _compute_metric_scores pour l'intégrer ici dans "compute"
        #
        # implémenter ici la logique de cache
        # pour le moment, ne mettre en cache que les prédictions, ne pas mettre en cache le résultat (à voir)
        #
        # -> cast(float, metric)
        #
        # integrer dans la clé de cache la signature de score_func, pour intégrer les partial
        # les kwargs etc ?

        results = _get_cached_response_values(
            cache=self.report._cache,
            estimator_hash=int(self.report._hash),
            estimator=self.report.estimator_,
            X=X,
            response_method=self.RESPONSE_METHOD,
            pos_label=pos_label,
            data_source=data_source,
            data_source_hash=data_source_hash,
        )

        for key_tuple, value, is_cached in results:
            if not is_cached:
                self.report._cache[key_tuple] = value

            if key_tuple[-1] != "predict_time":
                y_pred = value

        score = self.SCORE_FUNC(y, y_pred, **kwargs)

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
    def factory() -> Metric: ...


class Accuracy(Metric):
    NAME = "accuracy"
    VERBOSE_NAME = "Accuracy"
    SCORE_FUNC = accuracy_score
    RESPONSE_METHOD = "predict"
    GREATER_IS_BETTER = True
    CUSTOM = False

    @staticmethod
    def available(report) -> bool:
        return report.ml_task in ("binary-classification", "multiclass-classification")

    def __call__(
        self,
        *,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        data_source: DataSource = "test",
    ) -> float:
        return super().__call__(X=X, y=y, data_source=data_source)


Average = Literal["binary", "macro", "micro", "weighted", "samples"]


class Precision(Metric):
    NAME = "precision"
    VERBOSE_NAME = "Precision"
    SCORE_FUNC = precision_score
    RESPONSE_METHOD = "predict"
    GREATER_IS_BETTER = True
    CUSTOM = False

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
    NAME = "r2"
    VERBOSE_NAME = "R²"
    SCORE_FUNC = r2_score
    RESPONSE_METHOD = "predict"
    GREATER_IS_BETTER = True
    CUSTOM = False

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


# redefinir les metriques actuelles comme des Metric
# les register dynamiqument dans les init des reports si elles sont compatibles
#
# faire une factory qui acceptent des _base_scorer, des callables ou des nom de scorer scikit et qui transforme ça en Metric
# changer _compute_metric_scores pour l'intégrer ici dans "compute"

# case 1 : str -> find in scikit-learn the corresponding scorer
# case 2 : scorer -> scikit-learn scorer
# case 3 : callable -> create a custom scorer based on the provided callable (optional verbose_name + greater_is_better)

# https://github.com/scikit-learn/scikit-learn/blob/e9752287ffffecb2d2878ce1ed97a77a43941579/sklearn/metrics/_scorer.py#L798


# class AggregateCustomMetricFormula(CustomMetricFormula, ABC):  # or Protocol
#     aggregate: Aggregate = ("mean", "std")


class MetricRegistry:
    def __init__(self, report, /, *metric_classes):
        super().__init__()

        self.__report = report
        self.__metric_name_to_function = {}

        for metric_class in metric_classes:
            if (not metric_class.available(self.__report)):
                continue

            self.__metric_name_to_function[metric_class.NAME] = metric_class(
                self.__report
            )

    def __iter__(self):
        yield from self.__metrics_name_to_function.items()

    def __repr__(self):
        return str(self.__metric_name_to_function)

    def __getattr__(self, name):
        return self.__metric_name_to_function[name]

    def summary(self):
        raise NotImplementedError

    def append(self, _):
        # case 1 : str -> find in scikit-learn the corresponding scorer
        # case 2 : scorer -> scikit-learn scorer
        # case 3 : callable -> create a custom scorer based on the provided callable (optional verbose_name + greater_is_better)
        raise NotImplementedError


from skore import EstimatorReport


class NewEstimatorReport(EstimatorReport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # create registry and append default metrics
        self.metrics = MetricRegistry(
            self,
            Accuracy,
            Precision,
            R2
        )


# note important
# pour toutes les string de scorer commencant par "neg_" (genre "neg_rmse") -> on garde le greater_is_better,
# mais on change le signe pour n'avoir que des résultats positifs

# -------------------------


# def foobar(y_true, y_pred):
#     return 1


# report.registry.append(foobar, verbose_name="Ma métrique", greater_is_better=True)
