from concurrent.futures import ThreadPoolExecutor
from functools import partialmethod

from blake3 import blake3 as Blake3
from orjson import OPT_NON_STR_KEYS, OPT_SERIALIZE_NUMPY, dumps
from pydantic import ValidationError
from pytest import fixture, mark, param, raises

from skore_hub_project import Project, bytes_to_b64_str
from skore_hub_project.artifact.media import (
    Coefficients,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
)


def serialize(feature_importance) -> (bytes, str):
    if hasattr(feature_importance, "frame"):
        feature_importance = feature_importance.frame()

    feature_importance_bytes = dumps(
        feature_importance.fillna("NaN").to_dict(orient="tight"),
        option=(OPT_NON_STR_KEYS | OPT_SERIALIZE_NUMPY),
    )

    threads = 1 if (len(feature_importance_bytes) < 1e6) else Blake3.AUTO
    hasher = Blake3(max_threads=threads)
    checksum = hasher.update(feature_importance_bytes).digest()

    return feature_importance_bytes, f"blake3-{bytes_to_b64_str(checksum)}"


@fixture(autouse=True)
def monkeypatch_permutation(monkeypatch):
    import skore

    monkeypatch.setattr(
        "skore.EstimatorReport.feature_importance.permutation",
        partialmethod(
            skore.EstimatorReport.feature_importance.permutation,
            seed=42,
        ),
    )


@mark.parametrize(
    "Media,report,accessor,data_source",
    (
        param(
            PermutationTest,
            "binary_classification",
            "permutation",
            "test",
            id="PermutationTest",
        ),
        param(
            PermutationTrain,
            "binary_classification",
            "permutation",
            "train",
            id="PermutationTrain",
        ),
        param(
            MeanDecreaseImpurity,
            "binary_classification",
            "mean_decrease_impurity",
            None,
            id="MeanDecreaseImpurity",
        ),
        param(
            Coefficients,
            "regression",
            "coefficients",
            None,
            id="Coefficients",
        ),
        param(
            Coefficients,
            "cv_regression",
            "coefficients",
            None,
            id="Coefficients",
        ),
    ),
)
class TestFeatureImportance:
    def test_init_exception(self, Media, report, accessor, data_source, request):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        with raises(
            ValidationError,
            match=f"Input should be an instance of {report.__class__.__name__}",
        ):
            Media(project=project, report=None)

    def test_compute_available_accessor(
        self, Media, report, accessor, data_source, request
    ):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        function = getattr(report.feature_importance, accessor)
        function_kwargs = {"data_source": data_source} if data_source else {}
        content, _ = serialize(function(**function_kwargs))

        media = Media(project=project, report=report)
        media.compute()

        assert media.computed is True
        assert media.filepath.read_bytes() == content

    def test_compute_unavailable_accessor(
        self, monkeypatch, Media, report, accessor, data_source, request
    ):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        report.clear_cache()
        monkeypatch.delattr(report.feature_importance.__class__, accessor)

        media = Media(project=project, report=report)
        media.compute()

        assert media.computed is True
        assert media.filepath.stat().st_size == 0

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    @mark.respx()
    def test_upload_available_accessor(
        self, tmp_path, Media, report, accessor, data_source, upload_mock, request
    ):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        function = getattr(report.feature_importance, accessor)
        function_kwargs = {"data_source": data_source} if data_source else {}
        _, checksum = serialize(function(**function_kwargs))

        media = Media(project=project, report=report)

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
            "content_type": "application/vnd.dataframe",
            "pool": pool,
        }

    def test_upload_unavailable_accessor(
        self,
        monkeypatch,
        tmp_path,
        Media,
        report,
        accessor,
        data_source,
        upload_mock,
        request,
    ):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        report.clear_cache()
        monkeypatch.delattr(report.feature_importance.__class__, accessor)

        media = Media(project=project, report=report)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        assert media.computed is True
        assert media.uploaded is True

        # ensure that there is no residual file
        assert not len(list(tmp_path.iterdir()))

        # ensure `upload` is not called
        assert not upload_mock.called

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    @mark.respx()
    def test_model_dump(self, Media, report, accessor, data_source, request):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)

        function = getattr(report.feature_importance, accessor)
        function_kwargs = {"data_source": data_source} if data_source else {}
        _, checksum = serialize(function(**function_kwargs))

        media = Media(project=project, report=report)

        with ThreadPoolExecutor(max_workers=6) as pool:
            media.upload(pool=pool, checksums_being_uploaded=set())

        payload = media.model_dump()

        assert payload == {
            "content_type": "application/vnd.dataframe",
            "name": accessor,
            "data_source": data_source,
            "checksum": checksum,
        }

    def test_model_dump_exception(self, Media, report, accessor, data_source, request):
        project = Project("<tenant>", "<name>")
        report = request.getfixturevalue(report)
        media = Media(project=project, report=report)

        with raises(RuntimeError, match=r"Please use `artifact.upload\(\)` before"):
            media.model_dump()
