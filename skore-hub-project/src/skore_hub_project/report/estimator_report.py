from typing import Any, ClassVar

from pydantic import Field

from skore_hub_project.media import (
    Coefficients,
    EstimatorHtmlRepr,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
    # TableReportTest,
    # TableReportTrain,
)
from skore_hub_project.media.media import Media
from skore_hub_project.metric import (
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
from skore_hub_project.metric.metric import Metric
from skore_hub_project.report.report import ReportPayload

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
    MEDIAS: ClassVar[tuple[Media]] = (
        Coefficients,
        EstimatorHtmlRepr,
        MeanDecreaseImpurity,
        PermutationTest,
        PermutationTrain,
        # TableReportTest,
        # TableReportTrain,
    )

    report: EstimatorReport = Field(repr=False, exclude=True)
