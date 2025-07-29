from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from functools import partial
from math import isfinite
from typing import TYPE_CHECKING

dataclass = partial(dataclass, kw_only=True)

if TYPE_CHECKING:
    from typing import Final, Literal

    from skore import CrossValidationReport, EstimatorReport

    Report = CrossValidationReport | EstimatorReport


@dataclass
class Metric:
    report: InitVar[Report]
    value: float | None = field(init=False)
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool
    position: int | None = None

    @abstractmethod
    def __post_init__(self, report: Report): ...

    def __setattr__(self, name, value):
        if name == "value":
            with suppress(TypeError):
                if isfinite(value := float(value)):
                    return super().__setattr__(name, value)
            return super().__setattr__(name, None)
        return super().__setattr__(name, value)


@dataclass
class Accuracy(ABC, Metric):
    name: Final[str] = "accuracy"
    verbose_name: Final[str] = "Accuracy"
    greater_is_better: Final[bool] = True

    def __post_init__(self, report: EstimatorReport):
        self.value = (
            report.metrics.accuracy(data_source=self.data_source)
            if hasattr(report.metrics, "accuracy")
            else None
        )


@dataclass
class AccuracyTrain(Accuracy):
    data_source: Final[Literal["train"]] = "train"


@dataclass
class AccuracyTest(Accuracy):
    data_source: Final[Literal["test"]] = "test"


@dataclass
class AccuracyMean(ABC, Metric):
    name: Final[str] = "accuracy_mean"
    verbose_name: Final[str] = "Accuracy - MEAN"
    greater_is_better: Final[bool] = True

    def __post_init__(self, report: CrossValidationReport):
        df = report.metrics.accuracy(data_source=self.data_source)
        series = df[(report.estimator_name_, "mean")]

        self.value = series.iloc[0]


@dataclass
class AccuracyTrainMean(AccuracyMean):
    data_source: Final[Literal["train"]] = "train"


@dataclass
class AccuracyTestMean(AccuracyMean):
    data_source: Final[Literal["test"]] = "test"


@dataclass
class AccuracySTD(ABC, Metric):
    name: Final[str] = "accuracy_std"
    verbose_name: Final[str] = "Accuracy - STD"
    greater_is_better: Final[bool] = False

    def __post_init__(self, report: CrossValidationReport):
        df = report.metrics.accuracy(data_source=self.data_source)
        series = df[(report.estimator_name_, "std")]

        self.value = series.iloc[0]


@dataclass
class AccuracyTrainSTD(AccuracySTD):
    data_source: Final[Literal["train"]] = "train"


@dataclass
class AccuracyTestSTD(AccuracySTD):
    data_source: Final[Literal["test"]] = "test"
