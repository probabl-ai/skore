from pydantic import ValidationError
from pytest import fixture, mark, raises
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project import Project
from skore_hub_project.artifact.media import (
    EstimatorHtmlRepr,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    RocTest,
    RocTrain,
    TableReportTest,
    TableReportTrain,
)
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.metric import (
    AccuracyTest,
    AccuracyTrain,
    BrierScoreTest,
    BrierScoreTrain,
    FitTime,
    LogLossTest,
    LogLossTrain,
    PrecisionTest,
    PrecisionTrain,
    PredictTimeTest,
    PredictTimeTrain,
    RecallTest,
    RecallTrain,
    RocAucTest,
    RocAucTrain,
)
from skore_hub_project.report import EstimatorReportPayload


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    import io

    import joblib

    reports = [object] + getattr(object, "estimator_reports_", [])
    caches = [report_to_clear._cache for report_to_clear in reports]

    object.clear_cache()

    try:
        with io.BytesIO() as stream:
            joblib.dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in zip(reports, caches, strict=True):
            report._cache = cache

    with Serializer(pickle_bytes) as serializer:
        checksum = serializer.checksum

    return pickle_bytes, checksum


@fixture
def project():
    return Project("<tenant>", "<name>")


@fixture
def payload(project, binary_classification):
    # Force the compute of the permutations
    binary_classification.feature_importance.permutation(data_source="train", seed=42)
    binary_classification.feature_importance.permutation(data_source="test", seed=42)

    return EstimatorReportPayload(
        project=project,
        report=binary_classification,
        key="<key>",
    )


class TestEstimatorReportPayload:
    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    def test_pickle(
        self, binary_classification, project, payload, upload_mock, respx_mock
    ):
        pickle, checksum = serialize(binary_classification)

        # Ensure payload is well constructed
        assert payload.pickle.checksum == checksum

        # Ensure payload is well constructed
        assert payload.pickle.checksum == checksum

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "content": pickle,
            "content_type": "application/octet-stream",
        }

    def test_metrics(self, payload):
        assert list(map(type, payload.metrics)) == [
            AccuracyTest,
            AccuracyTrain,
            BrierScoreTest,
            BrierScoreTrain,
            LogLossTest,
            LogLossTrain,
            PrecisionTest,
            PrecisionTrain,
            RecallTest,
            RecallTrain,
            RocAucTest,
            RocAucTrain,
            FitTime,
            PredictTimeTest,
            PredictTimeTrain,
        ]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            EstimatorHtmlRepr,
            MeanDecreaseImpurity,
            PermutationTest,
            PermutationTrain,
            PrecisionRecallTest,
            PrecisionRecallTrain,
            RocTest,
            RocTrain,
            TableReportTest,
            TableReportTrain,
        ]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_model_dump(self, binary_classification, payload):
        _, checksum = serialize(binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("metrics")
        payload_dict.pop("medias")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": "35806b458ab1a6d0c675fd226d7fc34a",
            "ml_task": "binary-classification",
            "pickle": {
                "checksum": checksum,
                "content_type": "application/octet-stream",
            },
        }

    def test_exception(self):
        with raises(ValidationError):
            EstimatorReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
