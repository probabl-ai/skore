"""Class definition of the payload used to send an estimator report to ``hub``."""

from functools import cached_property
from typing import ClassVar, cast

from pydantic import Field, computed_field

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
    TableReportTest,
    TableReportTrain,
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

    METRICS: ClassVar[tuple[Metric, ...]] = cast(
        tuple[Metric, ...],
        (
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
        ),
    )
    MEDIAS: ClassVar[tuple[Media, ...]] = cast(
        tuple[Media, ...],
        (
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
        ),
    )

    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def parameters(self) -> EstimatorReportArtefact | dict:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned before being uploaded to the
        artefact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artefacts storage.

        .. deprecated
          The ``parameters`` property will be removed in favor of a new ``checksum``
          property in a near future.
        """
        return EstimatorReportArtefact(project=self.project, report=self.report)
