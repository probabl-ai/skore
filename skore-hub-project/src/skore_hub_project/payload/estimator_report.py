from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field, computed_field

from ..metric.accuracy import AccuracyTest, AccuracyTrain
from ..metric.brier_score import BrierScoreTest, BrierScoreTrain
from ..metric.log_loss import LogLossTest, LogLossTrain
from ..metric.precision import PrecisionTest, PrecisionTrain
from ..metric.r2 import R2Test, R2Train
from ..metric.recall import RecallTest, RecallTrain
from ..metric.rmse import RmseTest, RmseTrain
from ..metric.roc_auc import RocAucTest, RocAucTrain

from .report import ReportPayload

EstimatorReport = Any
Metric = Any


class EstimatorReportPayload(ReportPayload):
    METRICS: ClassVar[list[Metric]] = (
        AccuracyTrain,
        AccuracyTest,
        BrierScoreTrain,
        BrierScoreTest,
        LogLossTrain,
        LogLossTest,
        PrecisionTrain,
        PrecisionTest,
        R2Train,
        R2Test,
        RecallTrain,
        RecallTest,
        RmseTrain,
        RmseTest,
        RocAucTrain,
        RocAucTest,
        # # timings must be calculated last
        # timing("fit_time", "Fit time (s)", None, False, 1),
        # timing("predict_time", "Predict time (s)", "train", False, 2),
        # timing("predict_time", "Predict time (s)", "test", False, 2),
    )

    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
