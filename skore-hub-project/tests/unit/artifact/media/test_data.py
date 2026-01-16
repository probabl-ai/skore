from concurrent.futures import ThreadPoolExecutor
from json import load

from pydantic import ValidationError
from pytest import mark, param, raises

from skore_hub_project import Project
from skore_hub_project.artifact.media import TableReportTest, TableReportTrain


@mark.parametrize(
    "Media,data_source",
    (
        param(TableReportTest, "test", id="TableReportTest"),
        param(TableReportTrain, "train", id="TableReportTest"),
    ),
)
class TestTableReport:
    def test_init_exception(self, Media, data_source):
        project = Project("<tenant>", "<name>")

        with raises(
            ValidationError, match="Input should be an instance of EstimatorReport"
        ):
            Media(project=project, report=None)

    def test_compute(
        self, tmp_path, binary_classification, Media, data_source, upload_mock
    ):
        project = Project("<tenant>", "<name>")

        media = Media(project=project, report=binary_classification)
        media.compute()

        with media.filepath.open() as file:
            dataframe = load(file)

        assert media.computed is True
        assert {
            "n_rows",
            "n_columns",
            "n_constant_columns",
            "extract_head",
            "extract_tail",
            "columns",
            "top_associations",
        }.issubset(dataframe.keys())

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    @mark.respx()
    def test_upload(
        self, tmp_path, binary_classification, Media, data_source, upload_mock
    ):
        project = Project("<tenant>", "<name>")
        media = Media(project=project, report=binary_classification)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        assert media.computed is True
        assert media.uploaded is True

        # ensure that there is no residual file
        assert not len(list(tmp_path.iterdir()))

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs.pop("checksum")
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "filepath": media.filepath,
            "content_type": "application/vnd.skrub.table-report.v1+json",
            "pool": pool,
        }

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    @mark.respx()
    def test_model_dump(self, binary_classification, Media, data_source):
        project = Project("<tenant>", "<name>")
        media = Media(project=project, report=binary_classification)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        payload = media.model_dump()

        assert payload.pop("checksum")
        assert payload == {
            "content_type": "application/vnd.skrub.table-report.v1+json",
            "name": "table_report",
            "data_source": data_source,
        }

    def test_model_dump_exception(self, binary_classification, Media, data_source):
        project = Project("<tenant>", "<name>")
        media = Media(project=project, report=binary_classification)

        with raises(RuntimeError, match=r"Please use `artifact.upload\(\)` before"):
            media.model_dump()
