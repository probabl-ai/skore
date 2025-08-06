from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class Precision(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["precision"] = "precision"
    verbose_name: Literal["Precision"] = "Precision"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class PrecisionTrain(Precision):
    data_source: Literal["train"] = "train"


class PrecisionTest(Precision):
    data_source: Literal["test"] = "test"


class PrecisionMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["precision_mean"] = "precision_mean"
    verbose_name: Literal["Precision - MEAN"] = "Precision - MEAN"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class PrecisionTrainMean(PrecisionMean):
    data_source: Literal["train"] = "train"


class PrecisionTestMean(PrecisionMean):
    data_source: Literal["test"] = "test"


class PrecisionSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["precision_std"] = "precision_std"
    verbose_name: Literal["Precision - STD"] = "Precision - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.precision
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class PrecisionTrainSTD(PrecisionSTD):
    data_source: Literal["train"] = "train"


class PrecisionTestSTD(PrecisionSTD):
    data_source: Literal["test"] = "test"
