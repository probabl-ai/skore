from abc import ABC
from collections import UserList

import sklearn.metrics

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
    SCORE_FUNC: Callable  # foo(self, y_true, y_pred) -> float | Any
    RESPONSE_METHOD: str | Iterable[str]
    GREATER_IS_BETTER: bool
    CUSTOM: bool

    @staticmethod
    def available(estimator, ml_task) -> bool:
        raise NotImplementedError

    def compute(self, *, data_source="test", X=None, y=None) -> float | Any:
        # changer _compute_metric_scores pour l'intégrer ici dans "compute"
        #
        # implémenter ici la logique de cache
        # pour le moment, ne mettre en cache que les prédictions, ne pas mettre en cache le résultat (à voir)
        #
        # Il faudrait encapsuler `Scorer._score` et y ajouter la gestion du cache
        ...

    @staticmethod
    def factory() -> Metric: ...


class Accuracy(Metric):
    NAME = "accuracy"
    VERBOSE_NAME = "Accuracy"
    SCORE_FUNC = sklearn.metrics.accuracy_score
    RESPONSE_METHOD = "predict"
    GREATER_IS_BETTER = True
    CUSTOM = False

    @staticmethod
    def available(estimator, ml_task) -> bool:
        return hasattr(estimator, "_accuracy") and (
            ml_task in ("binary-classification", "multiclass-classification")
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


class AggregateCustomMetricFormula(CustomMetricFormula, ABC):  # or Protocol
    aggregate: Aggregate = ("mean", "std")


class MetricRegistry(UserList[type[Metric]]):
    def __init__(self, *, report: Report):
        super().__init__()

        self.__report = report

    def append(self, metric_cls: type[CustomMetricFormula]):
        # ensure metric_cls is valid
        # ensure metric_cls is not already present in the registry
        # append to the list
        # expose the tuple (name, compute) under the "report.metrics" accessor
        ...


class Report:
    def __init__(self):
        self.custom_metric_formula_registry: list[type[CustomMetricFormula]] = (
            CustomMetricFormulaRegistry(report=self)
        )


# note important
# pour toutes les string de scorer commencant par "neg_" (genre "neg_rmse") -> on garde le greater_is_better,
# mais on change le signe pour n'avoir que des résultats positifs

# -------------------------


def foobar(y_true, y_pred):
    return 1


report.registry.append(foobar, verbose_name="Ma métrique", greater_is_better=True)
