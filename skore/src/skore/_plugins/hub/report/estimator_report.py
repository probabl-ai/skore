"""Class definition of the payload used to send an estimator report to ``hub``."""

from functools import cached_property
from typing import ClassVar

from pydantic import computed_field

from skore import EstimatorReport
from skore._plugins.hub.artifact.media import (
    Coefficients,
    ConfusionMatrixDataFrameTestAll,
    ConfusionMatrixDataFrameTestNone,
    ConfusionMatrixDataFrameTrainAll,
    ConfusionMatrixDataFrameTrainNone,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PredictionErrorDataFrameTest,
    PredictionErrorDataFrameTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    TableReportTest,
    TableReportTrain,
)
from skore._plugins.hub.artifact.media.media import Media
from skore._plugins.hub.metric import Metric
from skore._plugins.hub.report.report import ReportPayload


class EstimatorReportPayload(ReportPayload[EstimatorReport]):
    """
    Payload used to send an estimator report to ``hub``.

    Attributes
    ----------
    MEDIAS : ClassVar[tuple[Media, ...]]
        The media classes that have to be computed from the report.
    project : Project
        The project to which the report payload should be sent.
    report : EstimatorReport
        The report on which to calculate the payload to be sent.
    key : str
        The key to associate to the report.
    """

    MEDIAS: ClassVar[tuple[type[Media[EstimatorReport]], ...]] = (
        Coefficients,
        ConfusionMatrixDataFrameTestAll,
        ConfusionMatrixDataFrameTestNone,
        ConfusionMatrixDataFrameTrainAll,
        ConfusionMatrixDataFrameTrainNone,
        EstimatorHtmlRepr,
        ImpurityDecrease,
        PermutationImportanceTest,
        PermutationImportanceTrain,
        PrecisionRecallDataFrameTest,
        PrecisionRecallDataFrameTrain,
        PredictionErrorDataFrameTest,
        PredictionErrorDataFrameTrain,
        RocDataFrameTest,
        RocDataFrameTrain,
        TableReportTest,
        TableReportTrain,
    )

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metrics(self) -> list[Metric[EstimatorReport]]:
        """
        The list of scalar metrics that have been computed from the report.

        Notes
        -----
        Unavailable metrics have been filtered out.

        All metrics whose value is not a scalar are currently ignored:
        - ignore ``list[float]`` for multi-output ML task,
        - ignore ``dict[str: float]`` for multi-classes ML task.
        """
        data = self.report.metrics.summarize(data_source="both").data
        scalar = data[
            data["label"].isna() & data["output"].isna() & data["average"].isna()
        ]

        return [
            Metric(
                name=row["metric_name"],
                verbose_name=row["metric_verbose_name"],
                data_source=row["data_source"],
                greater_is_better=row["greater_is_better"],
                value=row["score"],
            )
            for row in scalar.to_dict("records")
        ]
