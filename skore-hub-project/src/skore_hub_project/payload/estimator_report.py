from typing import Any
from functools import cached_property

from pydantic import Field, computed_field

from .report import ReportPayload


EstimatorReport = Any
Metric = Any


class EstimatorReportPayload(ReportPayload):
    report: EstimatorReport = Field(repr=False, exclude=True)

    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None
