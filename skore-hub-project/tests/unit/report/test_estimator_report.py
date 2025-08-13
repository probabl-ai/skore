from json import loads
from math import ceil
from operator import itemgetter
from urllib.parse import urljoin

import joblib
from httpx import Client, Response
from pytest import fixture, mark
from skore_hub_project import Project
from skore_hub_project.artefact.serializer import Serializer
from skore_hub_project.media import (
    Coefficients,
    EstimatorHtmlRepr,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
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
    R2Test,
    R2Train,
    RecallTest,
    RecallTrain,
    RmseTest,
    RmseTrain,
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
    monkeypatch.setattr(
        "skore_hub_project.project.project.HUBClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "skore_hub_project.artefact.upload.HUBClient",
        FakeClient,
    )


# test with regression
# test with binary_classification


def serialize(report) -> tuple[bytes, str, int]:
    cache = report._cache
    report._cache = {}

    try:
        with Serializer(report) as serializer:
            pickle = serializer.filepath.read_bytes()
            checksum = serializer.checksum
            chunk_size = ceil(len(pickle) / 2)
    finally:
        report._cache = cache

    return pickle, checksum, chunk_size


@mark.parametrize(
    "report,metrics,medias",
    (
        (
            "regression",
            [
                R2Test,
                R2Train,
                RmseTest,
                RmseTrain,
                FitTime,
                PredictTimeTest,
                PredictTimeTrain,
            ],
            [
                Coefficients,
                EstimatorHtmlRepr,
                PermutationTest,
                PermutationTrain,
                PredictionErrorTest,
                PredictionErrorTrain,
                TableReportTest,
                TableReportTrain,
            ],
        ),
        (
            "binary_classification",
            [
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
            ],
            [
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
            ],
        ),
    ),
)
def test_estimator_report_payload(
    monkeypatch, respx_mock, report, metrics, medias, request
):
    report = request.getfixturevalue(report)
    pickle, checksum, chunk_size = serialize(report)

    monkeypatch.setattr("skore_hub_project.artefact.upload.CHUNK_SIZE", chunk_size)
    respx_mock.post("projects/<tenant>/<name>/runs").mock(Response(200, json={"id": 0}))
    respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
        Response(
            200,
            json=[
                {"upload_url": "http://chunk2.com/", "chunk_id": 2},
                {"upload_url": "http://chunk1.com/", "chunk_id": 1},
            ],
        )
    )
    respx_mock.put("http://chunk1.com").mock(
        Response(200, headers={"etag": '"<etag1>"'})
    )
    respx_mock.put("http://chunk2.com").mock(
        Response(200, headers={"etag": '"<etag2>"'})
    )
    respx_mock.post("projects/<tenant>/<name>/artefacts/complete")

    project = Project("<tenant>", "<name>")
    payload = EstimatorReportPayload(project=project, report=report, key="<key>")
    payload_dict = payload.model_dump()
    requests = [call.request for call in respx_mock.calls]

    # Ensure upload is well done
    assert len(requests) == 5
    assert requests[0].url.path == "/projects/<tenant>/<name>/runs"
    assert requests[1].url.path == "/projects/<tenant>/<name>/artefacts"
    assert loads(requests[1].content.decode()) == [
        {
            "checksum": checksum,
            "chunk_number": 2,
            "content_type": "estimator-report",
        }
    ]
    assert sorted(
        (
            (str(requests[2].url), requests[2].content),
            (str(requests[3].url), requests[3].content),
        ),
        key=itemgetter(0),
    ) == [
        ("http://chunk1.com/", pickle[:chunk_size]),
        ("http://chunk2.com/", pickle[chunk_size:]),
    ]
    assert requests[4].url.path == "/projects/<tenant>/<name>/artefacts/complete"
    assert loads(requests[4].content.decode()) == [
        {
            "checksum": checksum,
            "etags": {
                "1": '"<etag1>"',
                "2": '"<etag2>"',
            },
        }
    ]

    # Ensure payload is well constructed
    assert list(map(type, payload.metrics)) == metrics
    assert list(map(type, payload.related_items)) == medias

    # Ensure payload dict is well constructed
    assert list(payload_dict.keys()) == [
        "key",
        "run_id",
        "estimator_class_name",
        "dataset_fingerprint",
        "ml_task",
        "parameters",
        "metrics",
        "related_items",
    ]

    assert payload_dict["run_id"] == 0
    assert payload_dict["parameters"] == {"checksum": checksum}
    assert payload_dict["estimator_class_name"] == report.estimator_name_
    assert payload_dict["dataset_fingerprint"] == joblib.hash(
        report.y_test if hasattr(report, "y_test") else report.y
    )
    assert payload_dict["ml_task"] == report.ml_task
