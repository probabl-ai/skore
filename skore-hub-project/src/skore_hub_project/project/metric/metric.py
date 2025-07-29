from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import Any, Literal, TYPE_CHECKING


dataclass = partial(dataclass, kw_only=True)

if TYPE_CHECKING:
    from skore import EstimatorReport, CrossValidationReport

    Report = EstimatorReport | CrossValidationReport


@dataclass
class Metric(ABC):
    report: InitVar[Report]
    value: float | None = field(init=False)
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool
    position: int | None = None

    @abstractmethod
    def __post_init__(self, report: Report): ...

    @property.setter
    def value(self, value: Any):
        with suppress(TypeError):
            if isfinite(value := float(value)):
                return value

        return None


@dataclass
class Accuracy(ABC, Metric):
    name: Final[str] = "accuracy"
    verbose_name: Final[str] = "Accuracy"
    greater_is_better: Final[str] = True

    def __post_init__(self, report: EstimatorReport):
        self.value = (
            report.metrics.accuracy(data_source=self.data_source)
            if hasattr(report, "metrics.accuracy")
            else None
        )


@dataclass
class AccuracyTrain(Metric):
    data_source: Final[str] = "train"


@dataclass
class AccuracyTest(Metric):
    data_source: Final[str] = "test"


@dataclass
class AccuracyMean(ABC, Metric):
    name: Final[str] = "accuracy_mean"
    verbose_name: Final[str] = "Accuracy - MEAN"
    greater_is_better: Final[str] = True

    def __post_init__(self, report: CrossValidationReport):
        df = report.metrics.accuracy(data_source=self.data_source)
        series = df[(report.estimator_name_, "mean")]

        self.value = series.iloc[0]


@dataclass
class AccuracyTrainMean(Metric):
    data_source: Final[str] = "train"


@dataclass
class AccuracyTestMean(Metric):
    data_source: Final[str] = "test"


@dataclass
class AccuracySTD(ABC, Metric):
    name: Final[str] = "accuracy_std"
    verbose_name: Final[str] = "Accuracy - STD"
    greater_is_better: Final[str] = False

    def __post_init__(self, report: CrossValidationReport):
        df = report.metrics.accuracy(data_source=self.data_source)
        series = df[(report.estimator_name_, "std")]

        self.value = series.iloc[0]


@dataclass
class AccuracyTrainSTD(Metric):
    data_source: Final[str] = "train"


@dataclass
class AccuracyTestSTD(Metric):
    data_source: Final[str] = "test"
