"""Class definition of the payload used to send a data category media to ``hub``."""

from typing import ClassVar, Literal

from pydantic import Field

from skore_hub_project import switch_mpl_backend
from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.serializer import JsonSerializer
from skore_hub_project.protocol import EstimatorReport


class TableReport(Media):  # noqa: D101
    serializer: ClassVar[type[JsonSerializer]] = JsonSerializer
    name: Literal["table_report"] = "table_report"
    media_type: Literal["application/vnd.skrub.table-report.v1+json"] = (
        "application/vnd.skrub.table-report.v1+json"
    )

    def object_to_upload(self) -> str:
        with switch_mpl_backend():
            display = (
                self.report.data.analyze()
                if self.data_source is None
                else self.report.data.analyze(data_source=self.data_source)
            )

        table_report = display.summary

        # Replace full dataset by its head/tail
        dataframe = table_report.pop("dataframe")
        table_report["extract_head"] = dataframe.head(3).to_dict(orient="split")
        table_report["extract_tail"] = dataframe.tail(3).to_dict(orient="split")

        # Remove irrelevant information
        del table_report["sample_table"]

        return table_report


class TableReportTrain(TableReport):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    data_source: Literal["train"] = "train"


class TableReportTest(TableReport):  # noqa: D101
    report: EstimatorReport = Field(repr=False, exclude=True)
    data_source: Literal["test"] = "test"
