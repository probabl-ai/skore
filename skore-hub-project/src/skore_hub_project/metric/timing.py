from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Literal, ClassVar

from pandas import DataFrame, Series
from pydantic import Field, computed_field

from .metric import Metric, cast_to_float

CrossValidationReport = Any
EstimatorReport = Any


class FitTime(Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["fit_time"] = "fit_time"
    verbose_name: Literal["Fit time (s)"] = "Fit time (s)"
    greater_is_better: Literal[False] = False
    position: Literal[1] = 1

    @computed_field
    @cached_property
    def value(self) -> float | None:
        timings: dict = self.report.metrics.timings()
        fit_time = timings.get("fit_time")

        return cast_to_float(fit_time)


class FitTimeAggregate(ABC, Metric):
    """
    Notes
    -----
    >>> report.metrics.timings()
                                mean       std
    Fit time (s)                 ...       ...
    Predict time test (s)        ...       ...
    Predict time train (s)       ...       ...
    """

    report: CrossValidationReport = Field(repr=False, exclude=True)
    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        timings: DataFrame = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            fit_times: Series = timings.loc[f"Fit time (s)"]
        except KeyError:
            return None

        return cast_to_float(fit_times.iloc[0])


class FitTimeMean(FitTimeAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["fit_time_mean"] = "fit_time_mean"
    verbose_name: Literal["Fit time (s) - MEAN"] = "Fit time (s) - MEAN"


class FitTimeStd(FitTimeAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["fit_time_std"] = "fit_time_std"
    verbose_name: Literal["Fit time (s) - STD"] = "Fit time (s) - STD"


class PredictTime(ABC, Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: Literal["predict_time"] = "predict_time"
    verbose_name: Literal["Predict time (s)"] = "Predict time (s)"
    greater_is_better: Literal[False] = False
    position: Literal[2] = 2

    @computed_field
    @cached_property
    def value(self) -> float | None:
        timings: dict = self.report.metrics.timings()
        predict_time = timings.get(f"predict_time_{self.data_source}")

        return cast_to_float(predict_time)


class PredictTimeTrain(PredictTime):
    data_source: Literal["train"] = "train"


class PredictTimeTest(PredictTime):
    data_source: Literal["test"] = "test"


class PredictTimeAggregate(ABC, Metric):
    """
    Notes
    -----
    >>> report.metrics.timings()
                                mean       std
    Fit time (s)                 ...       ...
    Predict time test (s)        ...       ...
    Predict time train (s)       ...       ...
    """

    report: CrossValidationReport = Field(repr=False, exclude=True)
    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: Literal[False] = False

    @computed_field
    @cached_property
    def value(self) -> float | None:
        timings: DataFrame = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            predict_times: Series = timings.loc[f"Predict time {self.data_source} (s)"]
        except KeyError:
            return None

        return cast_to_float(predict_times.iloc[0])


class PredictTimeTrainMean(PredictTimeAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["predict_time_mean"] = "predict_time_mean"
    verbose_name: Literal["Predict time (s) - MEAN"] = "Predict time (s) - MEAN"
    data_source: Literal["train"] = "train"


class PredictTimeTestMean(PredictTimeAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["predict_time_mean"] = "predict_time_mean"
    verbose_name: Literal["Predict time (s) - MEAN"] = "Predict time (s) - MEAN"
    data_source: Literal["test"] = "test"


class PredictTimeTrainStd(PredictTimeAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["predict_time_std"] = "predict_time_std"
    verbose_name: Literal["Predict time (s) - STD"] = "Predict time (s) - STD"
    data_source: Literal["train"] = "train"


class PredictTimeTestStd(PredictTimeAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["predict_time_std"] = "predict_time_std"
    verbose_name: Literal["Predict time (s) - STD"] = "Predict time (s) - STD"
    data_source: Literal["test"] = "test"
