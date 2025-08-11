from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class LogLoss(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.log_loss"]] = "metrics.log_loss"
    name: Literal["log_loss"] = "log_loss"
    verbose_name: Literal["Log loss"] = "Log loss"
    greater_is_better: Literal[False] = False
    position: Literal[4] = 4


class LogLossTrain(LogLoss):
    data_source: Literal["train"] = "train"


class LogLossTest(LogLoss):
    data_source: Literal["test"] = "test"


class LogLossMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.log_loss"]] = "metrics.log_loss"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["log_loss"] = "log_loss_mean"
    verbose_name: Literal["Log loss - MEAN"] = "Log loss - MEAN"
    greater_is_better: Literal[False] = False


class LogLossTrainMean(LogLossMean):
    data_source: Literal["train"] = "train"


class LogLossTestMean(LogLossMean):
    data_source: Literal["test"] = "test"


class LogLossStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.log_loss"]] = "metrics.log_loss"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["log_loss_std"] = "log_loss_std"
    verbose_name: Literal["Log loss - STD"] = "Log loss - STD"
    greater_is_better: Literal[False] = False


class LogLossTrainStd(LogLossStd):
    data_source: Literal["train"] = "train"


class LogLossTestStd(LogLossStd):
    data_source: Literal["test"] = "test"
