from json import loads
from urllib.parse import urljoin

from httpx import Client, Response
from pydantic import ValidationError
from pytest import fixture, mark, raises
from skore import EstimatorReport
from skore_hub_project import Project
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.media import (
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


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


@fixture(autouse=True)
def monkeypatch_client(monkeypatch):
    monkeypatch.setattr("skore_hub_project.project.project.HUBClient", FakeClient)
    monkeypatch.setattr("skore_hub_project.artifact.upload.HUBClient", FakeClient)


def serialize(object: EstimatorReport) -> tuple[bytes, str]:
    cache = object._cache
    object._cache = {}

    try:
        with Serializer(object) as serializer:
            pickle = serializer.filepath.read_bytes()
            checksum = serializer.checksum
    finally:
        object._cache = cache

    return pickle, checksum


@fixture
def payload(binary_classification):
    return EstimatorReportPayload(
        project=Project("<tenant>", "<name>"), report=binary_classification, key="<key>"
    )


@fixture
def monkeypatch_routes(respx_mock):
    respx_mock.post("projects/<tenant>/<name>/runs").mock(Response(200, json={"id": 0}))
    respx_mock.post("projects/<tenant>/<name>/artifacts").mock(
        Response(200, json=[{"upload_url": "http://chunk1.com/", "chunk_id": 1}])
    )
    respx_mock.put("http://chunk1.com").mock(
        Response(200, headers={"etag": '"<etag1>"'})
    )
    respx_mock.post("projects/<tenant>/<name>/artifacts/complete")


class TestEstimatorReportPayload:
    @mark.usefixtures("monkeypatch_routes")
    def test_parameters(self, binary_classification, payload, respx_mock):
        pickle, checksum = serialize(binary_classification)

        # Ensure payload dict is well constructed
        assert payload.parameters.checksum == checksum

        # Ensure upload is well done
        requests = [call.request for call in respx_mock.calls]

        assert len(requests) == 3
        assert requests[0].url.path == "/projects/<tenant>/<name>/artifacts"
        assert loads(requests[0].content.decode()) == [
            {
                "checksum": checksum,
                "chunk_number": 1,
                "content_type": "estimator-report",
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

    def test_related_items(self, payload):
        assert list(map(type, payload.related_items)) == [
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

    @mark.usefixtures("monkeypatch_routes")
    def test_model_dump(self, binary_classification, payload):
        _, checksum = serialize(binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("metrics")
        payload_dict.pop("related_items")

        assert payload_dict == {
            "key": "<key>",
            "run_id": 0,
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": "35806b458ab1a6d0c675fd226d7fc34a",
            "ml_task": "binary-classification",
            "parameters": {"checksum": checksum},
        }

    def test_exception(self):
        with raises(ValidationError):
            EstimatorReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
