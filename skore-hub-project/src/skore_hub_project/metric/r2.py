from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class R2(EstimatorReportMetric):
    accessor: ClassVar[str] = "metrics.r2"
    name: str = "r2"
    verbose_name: str = "R²"
    greater_is_better: bool = True


class R2Train(R2):
    data_source: Literal["train"] = "train"


class R2Test(R2):
    data_source: Literal["test"] = "test"


class R2Mean(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.r2"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: str = "r2_mean"
    verbose_name: str = "R² - MEAN"
    greater_is_better: bool = True


class R2TrainMean(R2Mean):
    data_source: Literal["train"] = "train"


class R2TestMean(R2Mean):
    data_source: Literal["test"] = "test"


class R2Std(CrossValidationReportMetric):
    accessor: ClassVar[str] = "metrics.r2"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: str = "r2_std"
    verbose_name: str = "R² - STD"
    greater_is_better: bool = False


class R2TrainStd(R2Std):
    data_source: Literal["train"] = "train"


class R2TestStd(R2Std):
    data_source: Literal["test"] = "test"
