from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field, computed_field

from ..metric.accuracy import AccuracyTest, AccuracyTrain
from .report import ReportPayload

EstimatorReport = Any
Metric = Any


class EstimatorReportPayload(ReportPayload):
    METRICS: ClassVar[list[Metric]] = (
        AccuracyTest,
        AccuracyTrain,
    )

    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
