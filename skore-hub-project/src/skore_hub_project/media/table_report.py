from .media import Media, Representation


EstimatorReport = Any


class TableReport(Media):
    report: EstimatorReport = Field(repr=False, exclude=True)
    key: Literal["table_report"] = "table_report"
    verbose_name: Literal["Table report"] = "Table report"
    category: Literal["data"] = "data"

    @computed_field
    @cached_property
    def representation(self) -> Representation:
        table_report_display = self.report.data.analyze(data_source=self.data_source)

        return Representation(
            media_type="application/vnd.skrub.table-report.v1+html",
            value=table_report_display._html_repr(),
        )


class TableReportTrain(Media):
    data_source: ClassVar[Literal["train"]]: "train"
    attributes: Literal[{"data_source": "train"}] = {"data_source": "train"}


class TableReportTest(Media):
    data_source: ClassVar[Literal["test"]]: "test"
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}

