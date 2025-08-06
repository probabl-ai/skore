from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class LogLoss(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["log_loss"] = "log_loss"
    verbose_name: Literal["Log loss"] = "Log loss"
    greater_is_better: Literal[True] = False
    position: Literal[4] = 4

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.log_loss
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class LogLossTrain(LogLoss):
    data_source: Literal["train"] = "train"


class LogLossTest(LogLoss):
    data_source: Literal["test"] = "test"


class LogLossMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["log_loss"] = "log_loss_mean"
    verbose_name: Literal["Log loss - MEAN"] = "Log loss - MEAN"
    greater_is_better: Literal[True] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.log_loss
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class LogLossTrainMean(LogLossMean):
    data_source: Literal["train"] = "train"


class LogLossTestMean(LogLossMean):
    data_source: Literal["test"] = "test"


class LogLossSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["log_loss_std"] = "log_loss_std"
    verbose_name: Literal["Log loss - STD"] = "Log loss - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.log_loss
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class LogLossTrainSTD(LogLossSTD):
    data_source: Literal["train"] = "train"


class LogLossTestSTD(LogLossSTD):
    data_source: Literal["test"] = "test"
