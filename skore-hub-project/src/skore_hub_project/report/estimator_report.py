from functools import cached_property
from typing import ClassVar

from pydantic import Field, computed_field
from skore import EstimatorReport

from skore_hub_project.artefact import EstimatorReportArtefact
from skore_hub_project.media import (
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
        PrecisionRecallTest,
        PrecisionRecallTrain,
        PredictionErrorTest,
        PredictionErrorTrain,
        RocTest,
        RocTrain,
        # TableReportTest,
        # TableReportTrain,
    )

    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field
    @cached_property
    def parameters(self) -> EstimatorReportArtefact | dict[()]:
        if self.upload:
            return EstimatorReportArtefact(project=self.project, report=self.report)
        return {}
