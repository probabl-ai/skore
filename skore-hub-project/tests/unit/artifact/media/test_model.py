from concurrent.futures import ThreadPoolExecutor

from blake3 import blake3 as Blake3
from pydantic import ValidationError
from pytest import mark, raises
from sklearn.utils import estimator_html_repr

from skore_hub_project import Project, bytes_to_b64_str
from skore_hub_project.artifact.media import EstimatorHtmlRepr


def serialize(estimator_html_repr) -> (bytes, str):
    estimator_html_repr_bytes = str.encode(estimator_html_repr, encoding="utf-8")

    threads = 1 if (len(estimator_html_repr_bytes) < 1e6) else Blake3.AUTO
    hasher = Blake3(max_threads=threads)
    checksum = hasher.update(estimator_html_repr_bytes).digest()

    return estimator_html_repr_bytes, f"blake3-{bytes_to_b64_str(checksum)}"


@mark.usefixtures("monkeypatch_artifact_hub_client")
@mark.usefixtures("monkeypatch_upload_routes")
@mark.usefixtures("monkeypatch_upload_with_mock")
class TestEstimatorHtmlRepr:
    def test_init_exception(self):
        project = Project("<tenant>", "<name>")

        with raises(
            ValidationError,
            match="Input should be an instance of EstimatorReport",
        ):
            EstimatorHtmlRepr(project=project, report=None)

    def test_compute(self, binary_classification):
        project = Project("<tenant>", "<name>")
        content, _ = serialize(estimator_html_repr(binary_classification.estimator_))

        media = EstimatorHtmlRepr(project=project, report=binary_classification)
        media.compute()

        assert media.computed is True
        assert media.filepath.read_bytes() == content

    def test_upload(self, tmp_path, binary_classification, upload_mock):
        project = Project("<tenant>", "<name>")
        _, checksum = serialize(estimator_html_repr(binary_classification.estimator_))

        media = EstimatorHtmlRepr(project=project, report=binary_classification)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        assert media.computed is True
        assert media.uploaded is True

        # ensure that there is no residual file
        assert not len(list(tmp_path.iterdir()))

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "filepath": media.filepath,
            "checksum": checksum,
            "content_type": "text/html",
            "pool": pool,
        }

    def test_model_dump(self, binary_classification):
        project = Project("<tenant>", "<name>")
        _, checksum = serialize(estimator_html_repr(binary_classification.estimator_))

        media = EstimatorHtmlRepr(project=project, report=binary_classification)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        payload = media.model_dump()

        assert payload == {
            "content_type": "text/html",
            "name": "estimator_html_repr",
            "data_source": None,
            "checksum": checksum,
        }

    def test_model_dump_exception(self, binary_classification):
        project = Project("<tenant>", "<name>")
        media = EstimatorHtmlRepr(project=project, report=binary_classification)

        with raises(RuntimeError, match=r"Please use `artifact.upload\(\)` before"):
            media.model_dump()
