from hashlib import blake2b
from json import loads

from pydantic import ValidationError
from pytest import mark, param, raises

from skore._plugins.hub.artifact.media import TableReportTest, TableReportTrain
from skore._plugins.hub.artifact.upload import plan_upload


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
    plan = plan_upload(media)

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

    # wrong type
    with raises(
        ValidationError, match="Input should be an instance of EstimatorReport"
    ):
        Media(project=project, report=None)
