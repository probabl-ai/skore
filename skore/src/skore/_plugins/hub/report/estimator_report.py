"""Class definition of the payload used to send an estimator report to ``hub``."""

from functools import cached_property
from typing import ClassVar

import pandas as pd
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

        Per-label (per-class) and per-output (multioutput regression) metrics are
        sent with their ``label``/``output`` dimension so the UI can expose a
        toggle. Metrics aggregated across labels or outputs (``average`` set) and
        non-scalar values (``NaN``) are still ignored.
        """
        data = self.report.metrics.summarize(data_source="both").data
        selected = data[data["average"].isna() & data["score"].notna()]

        return [
            Metric(
                name=row["metric_name"],
                verbose_name=row["metric_verbose_name"],
                data_source=row["data_source"],
                greater_is_better=row["greater_is_better"],
                value=row["score"],
                label=None if pd.isna(row["label"]) else row["label"],
                output=None if pd.isna(row["output"]) else int(row["output"]),
            )
            for row in selected.to_dict("records")
        ]
