from __future__ import annotations

from typing import Any, Literal, ClassVar

from .metric import EstimatorReportMetric, CrossValidationReportMetric


class R2(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.r2"]] = "metrics.r2"
    name: Literal["r2"] = "r2"
    verbose_name: Literal["R²"] = "R²"
    greater_is_better: Literal[True] = True


class R2Train(R2):
    data_source: Literal["train"] = "train"


class R2Test(R2):
    data_source: Literal["test"] = "test"


class R2Mean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.r2"]] = "metrics.r2"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["r2_mean"] = "r2_mean"
    verbose_name: Literal["R² - MEAN"] = "R² - MEAN"
    greater_is_better: Literal[True] = True


class R2TrainMean(R2Mean):
    data_source: Literal["train"] = "train"


class R2TestMean(R2Mean):
    data_source: Literal["test"] = "test"


class R2Std(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.r2"]] = "metrics.r2"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["r2_std"] = "r2_std"
    verbose_name: Literal["R² - STD"] = "R² - STD"
    greater_is_better: Literal[False] = False


class R2TrainStd(R2Std):
    data_source: Literal["train"] = "train"


class R2TestStd(R2Std):
    data_source: Literal["test"] = "test"
