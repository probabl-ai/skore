from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal

from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class BrierScore(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["brier_score"] = "brier_score"
    verbose_name: Literal["Brier score"] = "Brier score"
    greater_is_better: Literal[True] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.brier_score
        except AttributeError:
            return None
        else:
            return cast_to_float(function(data_source=self.data_source))


class BrierScoreTrain(BrierScore):
    data_source: Literal["train"] = "train"


class BrierScoreTest(BrierScore):
    data_source: Literal["test"] = "test"


class BrierScoreMean(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["brier_score_mean"] = "brier_score_mean"
    verbose_name: Literal["Brier score - MEAN"] = "Brier score - MEAN"
    greater_is_better: Literal[True] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.brier_score
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "mean")]

            return cast_to_float(series.iloc[0])


class BrierScoreTrainMean(BrierScoreMean):
    data_source: Literal["train"] = "train"


class BrierScoreTestMean(BrierScoreMean):
    data_source: Literal["test"] = "test"


class BrierScoreSTD(ABC, Metric):
    report: CrossValidationReport = Field(repr=False, exclude=True)
    name: Literal["brier_score_std"] = "brier_score_std"
    verbose_name: Literal["Brier score - STD"] = "Brier score - STD"
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        try:
            function = self.report.metrics.brier_score
        except AttributeError:
            return None
        else:
            df = function(data_source=self.data_source)
            series = df[(self.report.estimator_name_, "std")]

            return cast_to_float(series.iloc[0])


class BrierScoreTrainSTD(BrierScoreSTD):
    data_source: Literal["train"] = "train"


class BrierScoreTestSTD(BrierScoreSTD):
    data_source: Literal["test"] = "test"
