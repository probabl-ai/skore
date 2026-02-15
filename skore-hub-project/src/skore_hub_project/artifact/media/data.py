"""Definition of the payload used to associate a data category media with report."""

from typing import Literal

from skore_hub_project.artifact.media.media import Media, Report
from skore_hub_project.protocol import EstimatorReport


def _orjson_default(obj):
    """Default serializer for orjson to handle non-serializable types."""
    import pandas as pd

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class TableReport(Media[Report]):  # noqa: D101
    name: Literal["table_report"] = "table_report"
    data_source: Literal["train", "test"] | None = None
    content_type: Literal["application/vnd.skrub.table-report.v1+json"] = (
        "application/vnd.skrub.table-report.v1+json"
    )

    def content_to_upload(self) -> bytes:  # noqa: D102
        import orjson

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

        return orjson.dumps(
            table_report,
            option=(orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY),
            default=_orjson_default,
        )


class TableReportTrain(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["train"] = "train"


class TableReportTest(TableReport[EstimatorReport]):  # noqa: D101
    data_source: Literal["test"] = "test"
