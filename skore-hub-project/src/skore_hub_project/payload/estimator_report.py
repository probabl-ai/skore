from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field, computed_field

from ..metric import (
    AccuracyTest,
    AccuracyTrain,
    BrierScoreTest,
    BrierScoreTrain,
    FitTime,
    LogLossTest,
    LogLossTrain,
    PrecisionTest,
    PrecisionTrain,
    PredictTimeTest,
    PredictTimeTrain,
    R2Test,
    R2Train,
    RecallTest,
    RecallTrain,
    RmseTest,
    RmseTrain,
    RocAucTest,
    RocAucTrain,
)
from ..metric.metric import Metric
from .report import ReportPayload

EstimatorReport = Any


class EstimatorReportPayload(ReportPayload):
    METRICS: ClassVar[tuple[Metric]] = (
        AccuracyTest,
        AccuracyTrain,
        BrierScoreTest,
        BrierScoreTrain,
        LogLossTest,
        LogLossTrain,
        PrecisionTest,
        PrecisionTrain,
        R2Test,
        R2Train,
        RecallTest,
        RecallTrain,
        RmseTest,
        RmseTrain,
        RocAucTest,
        RocAucTrain,
        # timings must be calculated last
        FitTime,
        PredictTimeTrain,
        PredictTimeTest,
    )

    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
