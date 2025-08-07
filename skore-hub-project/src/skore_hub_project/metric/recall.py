from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class Recall(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.recall"]] = "metrics.recall"
    name: Literal["recall"] = "recall"
    verbose_name: Literal["Recall"] = "Recall"
    greater_is_better: Literal[True] = True


class RecallTrain(Recall):
    data_source: Literal["train"] = "train"


class RecallTest(Recall):
    data_source: Literal["test"] = "test"


class RecallMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.recall"]] = "metrics.recall"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["recall_mean"] = "recall_mean"
    verbose_name: Literal["Recall - MEAN"] = "Recall - MEAN"
    greater_is_better: Literal[True] = True


class RecallTrainMean(RecallMean):
    data_source: Literal["train"] = "train"


class RecallTestMean(RecallMean):
    data_source: Literal["test"] = "test"


class RecallStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.recall"]] = "metrics.recall"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["recall_std"] = "recall_std"
    verbose_name: Literal["Recall - STD"] = "Recall - STD"
    greater_is_better: Literal[False] = False


class RecallTrainStd(RecallStd):
    data_source: Literal["train"] = "train"


class RecallTestStd(RecallStd):
    data_source: Literal["test"] = "test"
