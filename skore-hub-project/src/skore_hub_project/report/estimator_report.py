"""Class definition of the payload used to send an estimator report to ``hub``."""

from typing import ClassVar

from pydantic import Field

from skore_hub_project.artifact.media import (
    Coefficients,
    EstimatorHtmlRepr,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
    TableReportTest,
    TableReportTrain,
)
from skore_hub_project.artifact.media.media import Media
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
from skore_hub_project.protocol import EstimatorReport
from skore_hub_project.report.report import ReportPayload


class EstimatorReportPayload(ReportPayload):
    """
    Payload used to send an estimator report to ``hub``.

    Attributes
    ----------
    METRICS : ClassVar[tuple[Metric, ...]]
        The metric classes that have to be computed from the report.
    MEDIAS : ClassVar[tuple[Media, ...]]
        The media classes that have to be computed from the report.
    project : Project
        The project to which the report payload should be sent.
    report : EstimatorReport
        The report on which to calculate the payload to be sent.
    key : str
        The key to associate to the report.
    """

    METRICS: ClassVar[tuple[type[Metric], ...]] = (
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
        PredictTimeTest,
        PredictTimeTrain,
    )
    MEDIAS: ClassVar[tuple[type[Media], ...]] = (
        Coefficients,
        EstimatorHtmlRepr,
        MeanDecreaseImpurity,
        PermutationTest,
        PermutationTrain,
        PrecisionRecallTest,
        PrecisionRecallTrain,
        PredictionErrorTest,
        PredictionErrorTrain,
        RocTest,
        RocTrain,
        TableReportTest,
        TableReportTrain,
    )

    report: EstimatorReport = Field(repr=False, exclude=True)
