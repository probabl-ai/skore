from __future__ import annotations

from abc import ABC
from typing import Literal, Any
from functools import cached_property

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
        return cast_to_float(
            self.report.metrics.accuracy(data_source=self.data_source)
            if hasattr(self.report.metrics, "accuracy")
            else None
        )


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
        df = self.report.metrics.accuracy(data_source=self.data_source)
        series = df[(self.report.estimator_name_, "mean")]

        return cast_to_float(series.iloc[0])


class AccuracyTrainMean(AccuracyMean):
    data_source: Literal["train"] = "train"


class AccuracyTestMean(AccuracyMean):
    data_source: Literal["test"] = "test"


class AccuracySTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        df = self.report.metrics.accuracy(data_source=self.data_source)
        series = df[(self.report.estimator_name_, "std")]

        return cast_to_float(series.iloc[0])


class AccuracyTrainSTD(AccuracySTD):
    data_source: Literal["train"] = "train"


class AccuracyTestSTD(AccuracySTD):
    data_source: Literal["test"] = "test"
