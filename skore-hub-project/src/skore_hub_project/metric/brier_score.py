from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class BrierScore(EstimatorReportMetric):
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    name: Literal["brier_score"] = "brier_score"
    verbose_name: Literal["Brier score"] = "Brier score"
    greater_is_better: Literal[False] = False


class BrierScoreTrain(BrierScore):
    data_source: Literal["train"] = "train"


class BrierScoreTest(BrierScore):
    data_source: Literal["test"] = "test"


class BrierScoreMean(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["brier_score_mean"] = "brier_score_mean"
    verbose_name: Literal["Brier score - MEAN"] = "Brier score - MEAN"
    greater_is_better: Literal[False] = False


class BrierScoreTrainMean(BrierScoreMean):
    data_source: Literal["train"] = "train"


class BrierScoreTestMean(BrierScoreMean):
    data_source: Literal["test"] = "test"


class BrierScoreStd(CrossValidationReportMetric):
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["brier_score_std"] = "brier_score_std"
    verbose_name: Literal["Brier score - STD"] = "Brier score - STD"
    greater_is_better: Literal[False] = False


class BrierScoreTrainStd(BrierScoreStd):
    data_source: Literal["train"] = "train"


class BrierScoreTestStd(BrierScoreStd):
    data_source: Literal["test"] = "test"
