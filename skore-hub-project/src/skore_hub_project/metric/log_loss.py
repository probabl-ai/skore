from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class LogLoss(EstimatorReportMetric):
    accessor: ClassVar[str] = "metrics.log_loss"
    name: str = "log_loss"
    verbose_name: str = "Log loss"
    greater_is_better: bool = False
    position: int = 4


class LogLossTrain(LogLoss):
    data_source: Literal["train"] = "train"


class LogLossTest(LogLoss):
    data_source: Literal["test"] = "test"


class LogLossMean(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.log_loss"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "log_loss_mean"
    verbose_name: str = "Log loss - MEAN"
    greater_is_better: bool = False


class LogLossTrainMean(LogLossMean):
    data_source: Literal["train"] = "train"


class LogLossTestMean(LogLossMean):
    data_source: Literal["test"] = "test"


class LogLossStd(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.log_loss"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "log_loss_std"
    verbose_name: str = "Log loss - STD"
    greater_is_better: bool = False


class LogLossTrainStd(LogLossStd):
    data_source: Literal["train"] = "train"


class LogLossTestStd(LogLossStd):
    data_source: Literal["test"] = "test"
