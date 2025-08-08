from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class Accuracy(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    name: Literal["accuracy"] = "accuracy"
    verbose_name: Literal["Accuracy"] = "Accuracy"
    greater_is_better: Literal[True] = True


class AccuracyTrain(Accuracy):
    data_source: Literal["train"] = "train"


class AccuracyTest(Accuracy):
    data_source: Literal["test"] = "test"


class AccuracyMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["accuracy_mean"] = "accuracy_mean"
    verbose_name: Literal["Accuracy - MEAN"] = "Accuracy - MEAN"
    greater_is_better: Literal[True] = True


class AccuracyTrainMean(AccuracyMean):
    data_source: Literal["train"] = "train"


class AccuracyTestMean(AccuracyMean):
    data_source: Literal["test"] = "test"


class AccuracyStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[True] = False


class AccuracyTrainStd(AccuracyStd):
    data_source: Literal["train"] = "train"


class AccuracyTestStd(AccuracyStd):
    data_source: Literal["test"] = "test"
