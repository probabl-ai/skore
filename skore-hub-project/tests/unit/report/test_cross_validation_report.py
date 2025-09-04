from json import loads
from urllib.parse import urljoin

from httpx import Client, Response
from pydantic import ValidationError
from pytest import fixture, mark, raises
from skore import CrossValidationReport
from skore_hub_project import Project
from skore_hub_project.artefact.serializer import Serializer
from skore_hub_project.media import (
    EstimatorHtmlRepr,
)
from skore_hub_project.metric import (
    AccuracyTestMean,
    AccuracyTestStd,
    AccuracyTrainMean,
    AccuracyTrainStd,
    BrierScoreTestMean,
    BrierScoreTestStd,
    BrierScoreTrainMean,
    BrierScoreTrainStd,
    FitTimeMean,
    FitTimeStd,
    LogLossTestMean,
    LogLossTestStd,
    LogLossTrainMean,
    LogLossTrainStd,
    PrecisionTestMean,
    PrecisionTestStd,
    PrecisionTrainMean,
    PrecisionTrainStd,
    PredictTimeTestMean,
    PredictTimeTestStd,
    PredictTimeTrainMean,
    PredictTimeTrainStd,
    RecallTestMean,
    RecallTestStd,
    RecallTrainMean,
    RecallTrainStd,
    RocAucTestMean,
    RocAucTestStd,
    RocAucTrainMean,
    RocAucTrainStd,
)
from skore_hub_project.report import (
    CrossValidationReportPayload,
    EstimatorReportPayload,
)


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
    monkeypatch.setattr("skore_hub_project.artefact.upload.HUBClient", FakeClient)


def serialize(object: CrossValidationReport) -> tuple[bytes, str, int]:
    reports = [object] + object.estimator_reports_
    caches = []

    for report in reports:
        caches.append(report._cache)
        report._cache = {}

    try:
        with Serializer(object) as serializer:
            pickle = serializer.filepath.read_bytes()
            checksum = serializer.checksum
    finally:
        for i, report in enumerate(reports):
            report._cache = caches[i]

    return pickle, checksum


@fixture
def payload(small_cv_binary_classification):
    return CrossValidationReportPayload(
        project=Project("<tenant>", "<name>"),
        report=small_cv_binary_classification,
        key="<key>",
    )


@fixture
def monkeypatch_routes(respx_mock):
    respx_mock.post("projects/<tenant>/<name>/runs").mock(Response(200, json={"id": 0}))
    respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
        Response(200, json=[{"upload_url": "http://chunk1.com/", "chunk_id": 1}])
    )
    respx_mock.put("http://chunk1.com").mock(
        Response(200, headers={"etag": '"<etag1>"'})
    )
    respx_mock.post("projects/<tenant>/<name>/artefacts/complete")


class TestCrossValidationReportPayload:
    def test_splitting_strategy_name(self, payload):
        assert payload.splitting_strategy_name == "StratifiedKFold"

    def test_splits(self, payload):
        assert payload.splits == [
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        ]

    def test_class_names(self, payload):
        assert payload.class_names == ["1", "0"]

    def test_classes(self, payload):
        assert payload.classes == [0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

    def test_estimators(self, payload, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )

        assert len(payload.estimators) == len(payload.report.estimator_reports_)

        for i, er_payload in enumerate(payload.estimators):
            assert isinstance(er_payload, EstimatorReportPayload)
            assert er_payload.report == payload.report.estimator_reports_[i]
            assert er_payload.upload is False
            assert er_payload.parameters == {}

    @mark.usefixtures("monkeypatch_routes")
    def test_parameters(self, small_cv_binary_classification, payload, respx_mock):
        pickle, checksum = serialize(small_cv_binary_classification)

        # Ensure payload dict is well constructed
        assert payload.parameters.checksum == checksum

        # Ensure upload is well done
        requests = [call.request for call in respx_mock.calls]

        assert len(requests) == 3
        assert requests[0].url.path == "/projects/<tenant>/<name>/artefacts"
        assert loads(requests[0].content.decode()) == [
            {
                "checksum": checksum,
                "chunk_number": 1,
                "content_type": "cross-validation-report",
            }
        ]
        assert requests[1].url == "http://chunk1.com/"
        assert requests[1].content == pickle
        assert requests[2].url.path == "/projects/<tenant>/<name>/artefacts/complete"
        assert loads(requests[2].content.decode()) == [
            {
                "checksum": checksum,
                "etags": {"1": '"<etag1>"'},
            }
        ]

    def test_metrics(self, payload):
        assert list(map(type, payload.metrics)) == [
            AccuracyTestMean,
            AccuracyTestStd,
            AccuracyTrainMean,
            AccuracyTrainStd,
            BrierScoreTestMean,
            BrierScoreTestStd,
            BrierScoreTrainMean,
            BrierScoreTrainStd,
            LogLossTestMean,
            LogLossTestStd,
            LogLossTrainMean,
            LogLossTrainStd,
            PrecisionTestMean,
            PrecisionTestStd,
            PrecisionTrainMean,
            PrecisionTrainStd,
            RecallTestMean,
            RecallTestStd,
            RecallTrainMean,
            RecallTrainStd,
            RocAucTestMean,
            RocAucTestStd,
            RocAucTrainMean,
            RocAucTrainStd,
            FitTimeMean,
            FitTimeStd,
            PredictTimeTestMean,
            PredictTimeTestStd,
            PredictTimeTrainMean,
            PredictTimeTrainStd,
        ]

    def test_related_items(self, payload):
        assert list(map(type, payload.related_items)) == [EstimatorHtmlRepr]

    @mark.usefixtures("monkeypatch_routes")
    def test_model_dump(self, small_cv_binary_classification, payload):
        _, checksum = serialize(small_cv_binary_classification)

        payload_dict = payload.model_dump()

        payload_dict.pop("estimators")
        payload_dict.pop("metrics")
        payload_dict.pop("related_items")

        assert payload_dict == {
            "key": "<key>",
            "run_id": 0,
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": "cffe9686d06a56d0afe0c3a29d3ac6bf",
            "ml_task": "binary-classification",
            "groups": None,
            "parameters": {"checksum": checksum},
            "splitting_strategy_name": "StratifiedKFold",
            "splits": [
                [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            ],
            "class_names": ["1", "0"],
            "classes": [0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        }

    def test_exception(self):
        with raises(ValidationError):
            CrossValidationReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
