from hashlib import blake2b
from json import loads

from pydantic import ValidationError
from pytest import mark, param, raises

from skore._plugins.hub.artifact.media import TableReportTest, TableReportTrain


@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTrain"),
    ),
)
@mark.respx()
def test_table_report(binary_classification, Media, data_source, project):
    media = Media(project=project, report=binary_classification)
    plan = media.local_plan()

    assert plan is not None
    assert plan.content_type == "application/vnd.skrub.table-report.v1+json"

    # ensure content is well constructed
    content_bytes = (
        plan.payload if isinstance(plan.payload, bytes) else plan.payload.read_bytes()
    )
    dataframe = loads(content_bytes)
    assert {
        "n_rows",
        "n_columns",
        "n_constant_columns",
        "extract_head",
        "extract_tail",
        "columns",
        "top_associations",
    }.issubset(dataframe.keys())

    assert plan.checksum == f"blake2b-{blake2b(content_bytes).hexdigest()}"

    # The model_dump of an artifact whose orchestrator hasn't run yet has
    # ``checksum=None`` (no ``_plan`` attached). Attach the plan to mirror
    # what ``ReportPayload.upload_artifacts`` does and assert the full shape.
    object.__setattr__(media, "_plan", plan)
    assert media.model_dump() == {
        "content_type": "application/vnd.skrub.table-report.v1+json",
        "name": "table_report",
        "data_source": data_source,
        "checksum": plan.checksum,
    }

    # wrong type
    with raises(
        ValidationError, match="Input should be an instance of EstimatorReport"
    ):
        Media(project=project, report=None)
