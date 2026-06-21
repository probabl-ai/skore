"""Definition of the payload used to associate a data category media with report."""

from typing import Any, Literal

import narwhals as nw

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.hub.artifact.media.media import Media, Report
from skore._plugins.hub.json import dumps


class TableReport(Media[Report]):  # noqa: D101
    name: Literal["table_report"] = "table_report"
    data_source: Literal["train", "test"] | None = None
    content_type: Literal["application/vnd.skrub.table-report.v1+json"] = (
        "application/vnd.skrub.table-report.v1+json"
    )

    def content_to_upload(self) -> bytes:  # noqa: D102
        display = (
            self.report.data.summarize()
            if (
                isinstance(self.report, CrossValidationReport)
                or (self.data_source is None)
            )
            else self.report.data.summarize(data_source=self.data_source)
        )

        table_report = display.summary

        # Replace full dataset by its head/tail
        dataframe_nw = nw.from_native(table_report.pop("dataframe"))

        def to_split(frame: nw.DataFrame[Any]) -> dict[str, Any]:
            return {
                "index": list(range(frame.shape[0])),
                "columns": frame.columns,
                "data": frame.rows(),
            }

        table_report["extract_head"] = to_split(dataframe_nw.head(3))
        table_report["extract_tail"] = to_split(dataframe_nw.tail(3))

        # Remove irrelevant information
        del table_report["sample_table"]

        return dumps(table_report)


class TableReportTrain(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["train"] = "train"


class TableReportTest(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["test"] = "test"
