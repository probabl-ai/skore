from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class Recall(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["recall"] = "recall"
    verbose_name: Literal["Recall"] = "Recall"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class RecallTrain(Recall):
    data_source: Literal["train"] = "train"


class RecallTest(Recall):
    data_source: Literal["test"] = "test"


class RecallMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["recall_mean"] = "recall_mean"
    verbose_name: Literal["Recall - MEAN"] = "Recall - MEAN"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class RecallTrainMean(RecallMean):
    data_source: Literal["train"] = "train"


class RecallTestMean(RecallMean):
    data_source: Literal["test"] = "test"


class RecallSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["recall_std"] = "recall_std"
    verbose_name: Literal["Recall - STD"] = "Recall - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.recall
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class RecallTrainSTD(RecallSTD):
    data_source: Literal["train"] = "train"


class RecallTestSTD(RecallSTD):
    data_source: Literal["test"] = "test"
