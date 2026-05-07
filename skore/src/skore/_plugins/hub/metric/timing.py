"""Class definition of the payload used to send a timing metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

from .metric import Metric, cast_to_float


class FitTime(Metric[EstimatorReport]):  # noqa: D101
    name: str = "fit_time"
    verbose_name: str = "Fit time (s)"
    greater_is_better: bool = False
    position: int = 1
    data_source: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        timings = self.report.metrics.timings()
        fit_time = timings.get("fit_time")

        self.value = cast_to_float(fit_time)


class FitTimeAggregate(Metric[CrossValidationReport]):  # noqa: D101
    # ``report.metrics.timings()``
    #
    #                             mean       std
    # Fit time (s)                 ...       ...
    # Predict time test (s)        ...       ...
    # Predict time train (s)       ...       ...

    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: bool = False
    data_source: None = None

    def compute(self) -> None:
        """Compute the value of the metric."""
        timings = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            fit_times = timings.loc["Fit time (s)"]
        except KeyError:
            self.value = None
        else:
            self.value = cast_to_float(fit_times.iloc[0])


class FitTimeMean(FitTimeAggregate):  # noqa: D101
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "fit_time_mean"
    verbose_name: str = "Fit time (s) - MEAN"
    position: int = 1


class FitTimeStd(FitTimeAggregate):  # noqa: D101
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "fit_time_std"
    verbose_name: str = "Fit time (s) - STD"
    position: None = None


class PredictTime(Metric[EstimatorReport]):  # noqa: D101
    name: str = "predict_time"
    verbose_name: str = "Predict time (s)"
    greater_is_better: bool = False
    position: int = 2

    def compute(self) -> None:
        """Compute the value of the metric."""
        timings = self.report.metrics.timings()
        predict_time = timings.get(f"predict_time_{self.data_source}")

        self.value = cast_to_float(predict_time)


class PredictTimeTrain(PredictTime):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictTimeTest(PredictTime):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictTimeAggregate(Metric[CrossValidationReport]):  # noqa: D101
    # ``report.metrics.timings()``
    #
    #                             mean       std
    # Fit time (s)                 ...       ...
    # Predict time test (s)        ...       ...
    # Predict time train (s)       ...       ...

    aggregate: ClassVar[Literal["mean", "std"]]
    greater_is_better: bool = False

    def compute(self) -> None:
        """Compute the value of the metric."""
        timings = self.report.metrics.timings(aggregate=self.aggregate)

        try:
            predict_times = timings.loc[f"Predict time {self.data_source} (s)"]
        except KeyError:
            self.value = None
        else:
            self.value = cast_to_float(predict_times.iloc[0])


class PredictTimeMean(PredictTimeAggregate):  # noqa: D101
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "predict_time_mean"
    verbose_name: str = "Predict time (s) - MEAN"
    position: int = 2


class PredictTimeTrainMean(PredictTimeMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictTimeTestMean(PredictTimeMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class PredictTimeStd(PredictTimeAggregate):  # noqa: D101
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "predict_time_std"
    verbose_name: str = "Predict time (s) - STD"
    position: None = None


class PredictTimeTrainStd(PredictTimeStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class PredictTimeTestStd(PredictTimeStd):  # noqa: D101
    data_source: Literal["test"] = "test"
