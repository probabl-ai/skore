from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class Accuracy(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["accuracy"] = "accuracy"
    verbose_name: Literal["Accuracy"] = "Accuracy"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.accuracy
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class AccuracyTrain(Accuracy):
    data_source: Literal["train"] = "train"


class AccuracyTest(Accuracy):
    data_source: Literal["test"] = "test"


class AccuracyMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["accuracy_mean"] = "accuracy_mean"
    verbose_name: Literal["Accuracy - MEAN"] = "Accuracy - MEAN"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.accuracy
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class AccuracyTrainMean(AccuracyMean):
    data_source: Literal["train"] = "train"


class AccuracyTestMean(AccuracyMean):
    data_source: Literal["test"] = "test"


class AccuracyStd(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.accuracy
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class AccuracyTrainStd(AccuracyStd):
    data_source: Literal["train"] = "train"


class AccuracyTestStd(AccuracyStd):
    data_source: Literal["test"] = "test"
