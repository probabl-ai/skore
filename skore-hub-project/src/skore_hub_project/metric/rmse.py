"""Class definition of the payload used to send a RMSE metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class Rmse(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.rmse"]] = "metrics.rmse"
    name: Literal["rmse"] = "rmse"
    verbose_name: Literal["RMSE"] = "RMSE"
    greater_is_better: Literal[False] = False
    position: Literal[3] = 3


class RmseTrain(Rmse):  # noqa: D101
    data_source: Literal["train"] = "train"


class RmseTest(Rmse):  # noqa: D101
    data_source: Literal["test"] = "test"


class RmseMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.rmse"]] = "metrics.rmse"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["rmse_mean"] = "rmse_mean"
    verbose_name: Literal["RMSE - MEAN"] = "RMSE - MEAN"
    greater_is_better: Literal[False] = False
    position: Literal[3] = 3


class RmseTrainMean(RmseMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class RmseTestMean(RmseMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class RmseStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.rmse"]] = "metrics.rmse"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["rmse_std"] = "rmse_std"
    verbose_name: Literal["RMSE - STD"] = "RMSE - STD"
    greater_is_better: Literal[False] = False


class RmseTrainStd(RmseStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class RmseTestStd(RmseStd):  # noqa: D101
    data_source: Literal["test"] = "test"
