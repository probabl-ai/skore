from __future__ import annotations

from typing import Any, Literal, ClassVar

from .metric import EstimatorReportMetric, CrossValidationReportMetric


class Precision(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.precision"]] = "metrics.precision"
    name: Literal["precision"] = "precision"
    verbose_name: Literal["Precision"] = "Precision"
    greater_is_better: Literal[True] = True


class PrecisionTrain(Precision):
    data_source: Literal["train"] = "train"


class PrecisionTest(Precision):
    data_source: Literal["test"] = "test"


class PrecisionMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.precision"]] = "metrics.precision"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["precision_mean"] = "precision_mean"
    verbose_name: Literal["Precision - MEAN"] = "Precision - MEAN"
    greater_is_better: Literal[True] = True


class PrecisionTrainMean(PrecisionMean):
    data_source: Literal["train"] = "train"


class PrecisionTestMean(PrecisionMean):
    data_source: Literal["test"] = "test"


class PrecisionStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.precision"]] = "metrics.precision"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["precision_std"] = "precision_std"
    verbose_name: Literal["Precision - STD"] = "Precision - STD"
    greater_is_better: Literal[False] = False


class PrecisionTrainStd(PrecisionStd):
    data_source: Literal["train"] = "train"


class PrecisionTestStd(PrecisionStd):
    data_source: Literal["test"] = "test"
