"""Class definition of the payload used to send a brier score metric to ``hub``."""

from __future__ import annotations

from typing import ClassVar, Literal

from .metric import CrossValidationReportMetric, EstimatorReportMetric


class BrierScore(EstimatorReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    name: Literal["brier_score"] = "brier_score"
    verbose_name: Literal["Brier score"] = "Brier score"
    greater_is_better: Literal[False] = False


class BrierScoreTrain(BrierScore):  # noqa: D101
    data_source: Literal["train"] = "train"


class BrierScoreTest(BrierScore):  # noqa: D101
    data_source: Literal["test"] = "test"


class BrierScoreMean(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    aggregate: ClassVar[Literal["mean"]] = "mean"
    name: Literal["brier_score_mean"] = "brier_score_mean"
    verbose_name: Literal["Brier score - MEAN"] = "Brier score - MEAN"
    greater_is_better: Literal[False] = False


class BrierScoreTrainMean(BrierScoreMean):  # noqa: D101
    data_source: Literal["train"] = "train"


class BrierScoreTestMean(BrierScoreMean):  # noqa: D101
    data_source: Literal["test"] = "test"


class BrierScoreStd(CrossValidationReportMetric):  # noqa: D101
    accessor: ClassVar[Literal["metrics.brier_score"]] = "metrics.brier_score"
    aggregate: ClassVar[Literal["std"]] = "std"
    name: Literal["brier_score_std"] = "brier_score_std"
    verbose_name: Literal["Brier score - STD"] = "Brier score - STD"
    greater_is_better: Literal[False] = False


class BrierScoreTrainStd(BrierScoreStd):  # noqa: D101
    data_source: Literal["train"] = "train"


class BrierScoreTestStd(BrierScoreStd):  # noqa: D101
    data_source: Literal["test"] = "test"
