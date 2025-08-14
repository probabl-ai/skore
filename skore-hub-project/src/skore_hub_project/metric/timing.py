from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Literal

from pandas import DataFrame, Series
from pydantic import Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from .metric import Metric, cast_to_float


class FitTime(Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: str = "fit_time"
    verbose_name: str = "Fit time (s)"
    greater_is_better: bool = False
    position: int = 1

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        timings: dict = self.report.metrics.timings()
        fit_time = timings.get("fit_time")

        return cast_to_float(fit_time)


class FitTimeAggregate(Metric):
    """
    Notes
    -----
    report.metrics.timings()

                                mean       std
    Fit time (s)                 ...       ...
    Predict time test (s)        ...       ...
    Predict time train (s)       ...       ...
    """

    report: CrossValidationReport = Field(repr=False, exclude=True)
    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        timings: DataFrame = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            fit_times: Series = timings.loc["Fit time (s)"]
        except KeyError:
            return None

        return cast_to_float(fit_times.iloc[0])


class FitTimeMean(FitTimeAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "fit_time_mean"
    verbose_name: str = "Fit time (s) - MEAN"
    position: int = 1


class FitTimeStd(FitTimeAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "fit_time_std"
    verbose_name: str = "Fit time (s) - STD"


class PredictTime(Metric):
    report: EstimatorReport = Field(repr=False, exclude=True)
    name: str = "predict_time"
    verbose_name: str = "Predict time (s)"
    greater_is_better: bool = False
    position: int = 2

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        timings: dict = self.report.metrics.timings()
        predict_time = timings.get(f"predict_time_{self.data_source}")

        return cast_to_float(predict_time)


class PredictTimeTrain(PredictTime):
    data_source: Literal["train"] = "train"


class PredictTimeTest(PredictTime):
    data_source: Literal["test"] = "test"


class PredictTimeAggregate(Metric):
    """
    Notes
    -----
    report.metrics.timings()

                                mean       std
    Fit time (s)                 ...       ...
    Predict time test (s)        ...       ...
    Predict time train (s)       ...       ...
    """

    report: CrossValidationReport = Field(repr=False, exclude=True)
    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: bool = False
    position: int = 2

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def value(self) -> float | None:
        timings: DataFrame = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            predict_times: Series = timings.loc[f"Predict time {self.data_source} (s)"]
        except KeyError:
            return None

        return cast_to_float(predict_times.iloc[0])


class PredictTimeMean(PredictTimeAggregate):
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "predict_time_mean"
    verbose_name: str = "Predict time (s) - MEAN"


class PredictTimeTrainMean(PredictTimeMean):
    data_source: Literal["train"] = "train"


class PredictTimeTestMean(PredictTimeMean):
    data_source: Literal["test"] = "test"


class PredictTimeStd(PredictTimeAggregate):
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "predict_time_std"
    verbose_name: str = "Predict time (s) - STD"


class PredictTimeTrainStd(PredictTimeStd):
    data_source: Literal["train"] = "train"


class PredictTimeTestStd(PredictTimeStd):
    data_source: Literal["test"] = "test"
