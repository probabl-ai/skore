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
from skore._plugins.hub.metric import EstimatorReportMetric
from skore._plugins.hub.report.report import ReportPayload
from skore._sklearn._plot.metrics.metrics_summary_display import MetricsSummaryRow


def _filter(row: MetricsSummaryRow) -> bool:
    """Condition to keep a metric row."""
    return (
        # Ignore non-float metrics for now
        (row["label"] is None and row["output"] is None and row["average"] is None)
        # Quick fix for a skore quirk: "fit_time" is computed for
        # data_source="test" even though that doesn't make sense,
        # because it needs to be shown in report.metrics.summarize().frame()
        # which by default has data_source="test"
        and not (row["metric_name"] == "fit_time" and row["data_source"] == "test")
    )


class EstimatorReportPayload(ReportPayload[EstimatorReport]):
    """
    Payload used to send an estimator report to ``hub``.

    Attributes
    ----------
    metrics : list[EstimatorReportMetric]
        The metrics that have to be computed from the report.
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
    def metrics(self) -> list[EstimatorReportMetric]:
        """
        The list of scalar metrics that have been computed from the report.

        Notes
        -----
        Unavailable metrics have been filtered out.

        All metrics whose value is not a scalar are currently ignored:
        - ignore ``list[float]`` for multi-output ML task,
        - ignore ``dict[str: float]`` for multi-classes ML task.

        The position field is used to drive the ``hub``'s parallel coordinates plot:
        - int [0, inf[, to be displayed at the position,
        - None, not to be displayed.
        """
        rows = self.report.metrics.summarize(data_source="both").rows
        positions = {name: i for i, name in enumerate(self.report._metric_registry)}
        return [
            EstimatorReportMetric(
                value=row["score"],
                report=self.report,
                name=row["metric_name"],
                verbose_name=row["metric_verbose_name"],
                data_source=row["data_source"],
                greater_is_better=row["greater_is_better"],
                position=positions[row["metric_name"]],
            )
            for row in rows
            if _filter(row)
        ]
