from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class RocAuc(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["roc_auc"] = "roc_auc"
    verbose_name: Literal["ROC AUC"] = "ROC AUC"
    greater_is_better: Literal[True] = True
    position: Literal[3] = 3

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.roc_auc
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class RocAucTrain(RocAuc):
    data_source: Literal["train"] = "train"


class RocAucTest(RocAuc):
    data_source: Literal["test"] = "test"


class RocAucMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["roc_auc_mean"] = "roc_auc_mean"
    verbose_name: Literal["ROC AUC - MEAN"] = "ROC AUC - MEAN"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.roc_auc
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class RocAucTrainMean(RocAucMean):
    data_source: Literal["train"] = "train"


class RocAucTestMean(RocAucMean):
    data_source: Literal["test"] = "test"


class RocAucSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["roc_auc_std"] = "roc_auc_std"
    verbose_name: Literal["ROC AUC - STD"] = "ROC AUC - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.roc_auc
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class RocAucTrainSTD(RocAucSTD):
    data_source: Literal["train"] = "train"


class RocAucTestSTD(RocAucSTD):
    data_source: Literal["test"] = "test"
