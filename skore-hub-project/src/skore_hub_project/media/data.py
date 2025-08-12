from functools import cached_property
from inspect import signature
from typing import Literal

from pydantic import Field, computed_field
from skore import EstimatorReport

from .media import Media, Representation


class TableReport(Media):
    report: EstimatorReport = Field(repr=False, exclude=True)
    key: str = "table_report"
    verbose_name: str = "Table report"
    category: Literal["data"] = "data"

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def representation(self) -> Representation:
        function = self.report.data.analyze
        function_parameters = signature(function).parameters
        function_kwargs = {
            k: v for k, v in self.attributes.items() if k in function_parameters
        }

        table_report_display = function(**function_kwargs)

        return Representation(
            media_type="application/vnd.skrub.table-report.v1+json",
            value=table_report_display._to_json(),
        )


class TableReportTrain(TableReport):
    attributes: dict = {"data_source": "train"}


class TableReportTest(TableReport):
    attributes: dict = {"data_source": "test"}
