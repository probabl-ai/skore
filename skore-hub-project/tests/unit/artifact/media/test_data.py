from json import loads

from pydantic import ValidationError
from pytest import mark, param, raises

from skore_hub_project.artifact.media import TableReportTest, TableReportTrain
from skore_hub_project.artifact.serializer import Serializer


@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTest"),
    ),
)
@mark.respx()
def test_table_report(binary_classification, Media, data_source, upload_mock, project):
    media = Media(project=project, report=binary_classification)
    media.model_dump()

    # ensure `upload` is well called
    assert upload_mock.called
    assert not upload_mock.call_args.args

    content = upload_mock.call_args.kwargs.pop("content")

    assert upload_mock.call_args.kwargs == {
        "project": project,
        "content_type": "application/vnd.skrub.table-report.v1+json",
    }

    with Serializer(content) as serializer:
        checksum = serializer.checksum

    # ensure content is well constructed
    dataframe = loads(content)

    assert {
        "n_rows",
        "n_columns",
        "n_constant_columns",
        "extract_head",
        "extract_tail",
        "columns",
        "top_associations",
    }.issubset(dataframe.keys())

    # ensure payload is well constructed
    assert media.model_dump() == {
        "content_type": "application/vnd.skrub.table-report.v1+json",
        "name": "table_report",
        "data_source": data_source,
        "checksum": checksum,
    }

    # wrong type
    with raises(
        ValidationError, match="Input should be an instance of EstimatorReport"
    ):
        Media(project=project, report=None)
