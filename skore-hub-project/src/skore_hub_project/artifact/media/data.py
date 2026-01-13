"""Definition of the payload used to associate a data category media with report."""

from typing import Literal

from orjson import OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY, dumps

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import EstimatorReport


class TableReport(Media[Report]):  # noqa: D101
    name: Literal["table_report"] = "table_report"
    data_source: Literal["train", "test"] | None = None
    content_type: Literal["application/vnd.skrub.table-report.v1+json"] = (
        "application/vnd.skrub.table-report.v1+json"
    )

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

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

        self.computed = True
        self.filepath.write_bytes(
            dumps(
                table_report,
                option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
            )
        )


class TableReportTrain(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["train"] = "train"


class TableReportTest(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["test"] = "test"
