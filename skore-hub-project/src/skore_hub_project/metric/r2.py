from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class R2(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["r2"] = "r2"
    verbose_name: Literal["R²"] = "R²"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.r2
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class R2Train(R2):
    data_source: Literal["train"] = "train"


class R2Test(R2):
    data_source: Literal["test"] = "test"


class R2Mean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["r2_mean"] = "r2_mean"
    verbose_name: Literal["R² - MEAN"] = "R² - MEAN"
    greater_is_better: Literal[True] = True

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.r2
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class R2TrainMean(R2Mean):
    data_source: Literal["train"] = "train"


class R2TestMean(R2Mean):
    data_source: Literal["test"] = "test"


class R2STD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["r2_std"] = "r2_std"
    verbose_name: Literal["R² - STD"] = "R² - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.r2
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class R2TrainSTD(R2STD):
    data_source: Literal["train"] = "train"


class R2TestSTD(R2STD):
    data_source: Literal["test"] = "test"
