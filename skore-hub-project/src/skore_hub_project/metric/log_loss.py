"""Class definition of the payload used to send a log loss metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class LogLoss(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.log_loss"
    name: str = "log_loss"
    verbose_name: str = "Log loss"
    greater_is_better: bool = False
    position: int = 4


class LogLossTrain(LogLoss):  # noqa: D101
    data_source: Literal["train"] = "train"


class LogLossTest(LogLoss):  # noqa: D101
    data_source: Literal["test"] = "test"


class LogLossMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.log_loss"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "log_loss_mean"
    verbose_name: str = "Log loss - MEAN"
    greater_is_better: bool = False
    position: int = 4


class LogLossTrainMean(LogLossMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class LogLossTestMean(LogLossMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class LogLossStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[str] = "metrics.log_loss"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "log_loss_std"
    verbose_name: str = "Log loss - STD"
    greater_is_better: bool = False


class LogLossTrainStd(LogLossStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class LogLossTestStd(LogLossStd):  # noqa: D101
    data_source: Literal["test"] = "test"
