"""Class definition of the payload used to send an accuracy metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class Accuracy(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    name: Literal["accuracy"] = "accuracy"
    verbose_name: Literal["Accuracy"] = "Accuracy"
    greater_is_better: Literal[True] = True
    position: None = None


class AccuracyTrain(Accuracy):  # noqa: D101
    data_source: Literal["train"] = "train"


class AccuracyTest(Accuracy):  # noqa: D101
    data_source: Literal["test"] = "test"


class AccuracyMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["accuracy_mean"] = "accuracy_mean"
    verbose_name: Literal["Accuracy - MEAN"] = "Accuracy - MEAN"
    greater_is_better: Literal[True] = True
    position: None = None


class AccuracyTrainMean(AccuracyMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class AccuracyTestMean(AccuracyMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class AccuracyStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.accuracy"]] = "metrics.accuracy"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["accuracy_std"] = "accuracy_std"
    verbose_name: Literal["Accuracy - STD"] = "Accuracy - STD"
    greater_is_better: Literal[False] = False
    position: None = None


class AccuracyTrainStd(AccuracyStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class AccuracyTestStd(AccuracyStd):  # noqa: D101
    data_source: Literal["test"] = "test"
