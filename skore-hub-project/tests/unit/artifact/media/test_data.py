from json import load

from pydantic import ValidationError
from pytest import mark, param, raises

from skore_hub_project.artifact.media import TableReportTest, TableReportTrain


@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTest"),
    ),
)
class TestTableReport:
    def test_init_exception(self, Media, data_source, project):
        with raises(
            ValidationError, match="Input should be an instance of EstimatorReport"
        ):
            Media(project=project, report=None)

    def test_compute(
        self, binary_classification, Media, data_source, upload_mock, project
    ):
        media = Media(project=project, report=binary_classification)

        assert media.computed is False

        media.compute()

        assert media.computed is True

        with media.filepath.open() as file:
            dataframe = load(file)

        assert {
            "n_rows",
            "n_columns",
            "n_constant_columns",
            "extract_head",
            "extract_tail",
            "columns",
            "top_associations",
        }.issubset(dataframe.keys())

    @mark.respx()
    def test_upload(
        self, binary_classification, Media, data_source, upload_mock, project
    ):
        media = Media(project=project, report=binary_classification)

        assert media.uploaded is False

        media.compute()
        media.upload()

        assert media.uploaded is True

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs.pop("checksum")
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "filepath": media.filepath,
            "content_type": "application/vnd.skrub.table-report.v1+json",
        }

    @mark.respx()
    def test_model_dump(self, binary_classification, Media, data_source, project):
        media = Media(project=project, report=binary_classification)

        media.compute()
        media.upload()

        payload = media.model_dump()

        assert payload.pop("checksum")
        assert payload == {
            "content_type": "application/vnd.skrub.table-report.v1+json",
            "name": "table_report",
            "data_source": data_source,
        }
