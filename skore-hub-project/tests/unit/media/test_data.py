from pydantic import ValidationError
from pytest import mark, param, raises
from skore_hub_project.media import TableReportTest, TableReportTrain


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

    representation_value = media_dict["representation"]["value"]
    media_dict["representation"]["value"] = {}
    assert media_dict == {
        "key": "table_report",
        "verbose_name": "Table report",
        "category": "data",
        "attributes": {"data_source": data_source},
        "parameters": {},
        "representation": {
            "media_type": "application/vnd.skrub.table-report.v1+json",
            "value": {},
        },
    }
    assert set(
        [
            "n_rows",
            "n_columns",
            "n_constant_columns",
            "extract",
            "columns",
            "top_associations",
        ]
    ).issubset(representation_value.keys())

    # wrong type
    with raises(ValidationError):
        Media(report=None)
