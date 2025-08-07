from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class RocAuc(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.roc_auc"]] = "metrics.roc_auc"
    name: Literal["roc_auc"] = "roc_auc"
    verbose_name: Literal["ROC AUC"] = "ROC AUC"
    greater_is_better: Literal[True] = True
    position: Literal[3] = 3


class RocAucTrain(RocAuc):
    data_source: Literal["train"] = "train"


class RocAucTest(RocAuc):
    data_source: Literal["test"] = "test"


class RocAucMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.roc_auc"]] = "metrics.roc_auc"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["roc_auc_mean"] = "roc_auc_mean"
    verbose_name: Literal["ROC AUC - MEAN"] = "ROC AUC - MEAN"
    greater_is_better: Literal[True] = True


class RocAucTrainMean(RocAucMean):
    data_source: Literal["train"] = "train"


class RocAucTestMean(RocAucMean):
    data_source: Literal["test"] = "test"


class RocAucStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.roc_auc"]] = "metrics.roc_auc"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["roc_auc_std"] = "roc_auc_std"
    verbose_name: Literal["ROC AUC - STD"] = "ROC AUC - STD"
    greater_is_better: Literal[False] = False


class RocAucTrainStd(RocAucStd):
    data_source: Literal["train"] = "train"


class RocAucTestStd(RocAucStd):
    data_source: Literal["test"] = "test"
