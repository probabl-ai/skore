from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pandas import DataFrame
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


class AccuracyAggregate(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    aggregate: ClassVar[Literal["mean", "std"]]

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.accuracy
        except AttributeError:
            return None
        else:
            accuracies: DataFrame = function(
                data_source=self.data_source,
                aggregate=self.aggregate,
            )

            return cast_to_float(accuracies.iloc[0, 0])


class AccuracyTrainMean(AccuracyAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["accuracy_mean"] = "accuracy_mean"
    verbose_name: Literal["Accuracy - MEAN"] = "Accuracy - MEAN"
    greater_is_better: Literal[True] = True
    data_source: Literal["train"] = "train"


class AccuracyTestMean(AccuracyAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["accuracy_mean"] = "accuracy_mean"
    verbose_name: Literal["Accuracy - MEAN"] = "Accuracy - MEAN"
    greater_is_better: Literal[True] = True
    data_source: Literal["test"] = "test"


class AccuracyTrainStd(AccuracyAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[True] = False
    data_source: Literal["train"] = "train"


class AccuracyTestStd(AccuracyAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[True] = False
    data_source: Literal["test"] = "test"
