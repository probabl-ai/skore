from dataclasses import dataclass, KW_ONLY, field, InitVar
from typing import Any, Callable, Literal
from functools import partial


def call(report, accessor, **kwargs) -> float | None:
    if hasattr(report, accessor):
        value = getattr(report, accessor)(**kwargs)

        with suppress(TypeError):
            if isfinite(value := float(value)):
                return value

    return None


@dataclass(kw_only=True)
class Metric:
    report: InitVar[Report]
    function: InitVar[Callable[[EstimatorReport], float]]
    value: float | None = field(init=False)
    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None = None
    greater_is_better: bool
    position: int | None = None

    def __post_init__(
        self, report: EstimatorReport, function: Callable[[Report], float]
    ):
        self.value = function(report)  # or None


@dataclass(kw_only=True)
class Accuracy(Metric):
    name: Final[str] = "accuracy"
    verbose_name: Final[str] = "Accuracy"
    greater_is_better: Final[str] = True


@dataclass(kw_only=True)
class AccuracyTrain(Metric):
    data_source: Final[str] = "train"
    function: InitVar[Final[Callable[[EstimatorReport], float]]] = partial(
        call,
        accessor="metrics.accuracy",
        kwargs={"data_source": "train"},
    )


@dataclass(kw_only=True)
class AccuracyTest(Metric):
    data_source: Final[str] = "test"
    function: InitVar[Final[Callable[[EstimatorReport], float]]] = partial(
        call,
        accessor="metrics.accuracy",
        kwargs={"data_source": "test"},
    )


@dataclass(kw_only=True)
class AccuracyMean(Metric):
    @staticmethod
    def compute(report, data_source) -> float:
        df = report.metrics.accuracy(data_source=data_source)
        series = df[(report.estimator_name_, "mean")]

        return float(series.iloc[0])

    name: Final[str] = "accuracy_mean"
    verbose_name: Final[str] = "Accuracy - MEAN"
    greater_is_better: Final[str] = True


@dataclass(kw_only=True)
class AccuracyTrainMean(Metric):
    data_source: Final[str] = "train"
    function: InitVar[Final[Callable[[CrossValidationReport], float]]] = partial(
        AccuracyMean.compute, data_source="train"
    )


@dataclass(kw_only=True)
class AccuracyTestMean(Metric):
    data_source: Final[str] = "test"
    function: InitVar[Final[Callable[[CrossValidationReport], float]]] = partial(
        AccuracyMean.compute, data_source="test"
    )


@dataclass(kw_only=True)
class AccuracySTD(Metric):
    @staticmethod
    def compute(report, data_source) -> float:
        df = report.metrics.accuracy(data_source=data_source)
        series = df[(report.estimator_name_, "std")]

        return float(series.iloc[0])

    name: Final[str] = "accuracy_std"
    verbose_name: Final[str] = "Accuracy - STD"
    greater_is_better: Final[str] = False


@dataclass(kw_only=True)
class AccuracyTrainSTD(Metric):
    data_source: Final[str] = "train"
    function: InitVar[Final[Callable[[CrossValidationReport], float]]] = partial(
        AccuracySTD.compute, data_source="train"
    )


@dataclass(kw_only=True)
class AccuracyTestSTD(Metric):
    data_source: Final[str] = "test"
    function: InitVar[Final[Callable[[CrossValidationReport], float]]] = partial(
        AccuracySTD.compute, data_source="test"
    )
