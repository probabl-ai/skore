from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class Precision(EstimatorReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    name: str = "precision"
    verbose_name: str = "Precision"
    greater_is_better: bool = True


class PrecisionTrain(Precision):
    data_source: Literal["train"] = "train"


class PrecisionTest(Precision):
    data_source: Literal["test"] = "test"


class PrecisionMean(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "precision_mean"
    verbose_name: str = "Precision - MEAN"
    greater_is_better: bool = True


class PrecisionTrainMean(PrecisionMean):
    data_source: Literal["train"] = "train"


class PrecisionTestMean(PrecisionMean):
    data_source: Literal["test"] = "test"


class PrecisionStd(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.precision"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "precision_std"
    verbose_name: str = "Precision - STD"
    greater_is_better: bool = False


class PrecisionTrainStd(PrecisionStd):
    data_source: Literal["train"] = "train"


class PrecisionTestStd(PrecisionStd):
    data_source: Literal["test"] = "test"
