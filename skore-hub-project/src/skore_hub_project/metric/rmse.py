from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class Rmse(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["rmse"] = "rmse"
    verbose_name: Literal["RMSE"] = "RMSE"
    greater_is_better: Literal[True] = False
    position: Literal[3] = 3

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.rmse
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class RmseTrain(Rmse):
    data_source: Literal["train"] = "train"


class RmseTest(Rmse):
    data_source: Literal["test"] = "test"


class RmseMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["rmse_mean"] = "rmse_mean"
    verbose_name: Literal["RMSE - MEAN"] = "RMSE - MEAN"
    greater_is_better: Literal[True] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.rmse
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class RmseTrainMean(RmseMean):
    data_source: Literal["train"] = "train"


class RmseTestMean(RmseMean):
    data_source: Literal["test"] = "test"


class RmseSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["rmse_std"] = "rmse_std"
    verbose_name: Literal["RMSE - STD"] = "RMSE - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.rmse
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class RmseTrainSTD(RmseSTD):
    data_source: Literal["train"] = "train"


class RmseTestSTD(RmseSTD):
    data_source: Literal["test"] = "test"
