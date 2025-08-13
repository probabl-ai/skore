from pydantic import ValidationError
from pytest import fixture, mark, param, raises
from skore_hub_project.media import TableReportTest, TableReportTrain


@fixture(autouse=True)
def monkeypatch_to_json(monkeypatch):
    monkeypatch.setattr(
        "skore._sklearn._plot.TableReportDisplay._to_json", lambda self: "[0,1]"
    )


@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTest"),
    ),
)
def test_table_report(binary_classification, Media, data_source):
    media = Media(report=binary_classification)
    media_dict = media.model_dump()

    assert media_dict == {
        "key": "table_report",
        "verbose_name": "Table report",
        "category": "data",
        "attributes": {"data_source": data_source},
        "parameters": {},
        "representation": {
            "media_type": "application/vnd.skrub.table-report.v1",
            "value": [0, 1],
        },
    }

    # wrong type
    with raises(ValidationError):
        Media(report=None)
