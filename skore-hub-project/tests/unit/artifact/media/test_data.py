from httpx import Response
from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project import Project
from skore_hub_project.artifact.media import TableReportTest, TableReportTrain


@mark.usefixtures("monkeypatch_artifact_hub_client")
@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTest"),
    ),
)
def test_table_report(respx_mock, binary_classification, Media, data_source):
    respx_mock.post("projects/<tenant>/<name>/artifacts").mock(
        Response(200, json=[{"upload_url": "http://chunk1.com/", "chunk_id": 1}])
    )
    respx_mock.put("http://chunk1.com").mock(
        Response(200, headers={"etag": '"<etag1>"'})
    )
    respx_mock.post("projects/<tenant>/<name>/artifacts/complete")

    project = Project("<tenant>", "<name>")
    media = Media(project=project, report=binary_classification)
    media_dict = media.model_dump()

    # ensure table report is well uploaded
    requests = [call.request for call in respx_mock.calls]

    assert len(requests) == 3
    assert requests[0].url.path == "/projects/<tenant>/<name>/artifacts"
    assert loads(requests[0].content.decode()) == [
        {
            "checksum": checksum,
            "chunk_number": 1,
            "content_type": "application/vnd.skrub.table-report.v1+json",
        }
    ]
    assert requests[1].url == "http://chunk1.com/"
    assert requests[1].content == pickle
    assert requests[2].url.path == "/projects/<tenant>/<name>/artifacts/complete"
    assert loads(requests[2].content.decode()) == [
        {
            "checksum": checksum,
            "etags": {"1": '"<etag1>"'},
        }
    ]

    # ensure payload is well constructed

    assert media_dict == {
        "content_type": "application/vnd.skrub.table-report.v1+json",
        "name": "table_report",
        "data_source": data_source,
        "checksum": "blake3-X4hLmhEe9H+ucmbL6yZ1WuhIn1HuAFEIcO1AFEvxkG0=",
    }

    representation_value = media_dict["representation"]["value"]
    assert set(
        [
            "n_rows",
            "n_columns",
            "n_constant_columns",
            "extract_head",
            "extract_tail",
            "columns",
            "top_associations",
        ]
    ).issubset(representation_value.keys())

    # wrong type
    with raises(ValidationError):
        Media(report=None)
