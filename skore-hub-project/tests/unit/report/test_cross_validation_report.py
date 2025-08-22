from json import loads
from urllib.parse import urljoin

import numpy as np
from httpx import Client, Response
from pydantic import ValidationError
from pytest import fixture, mark, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, ShuffleSplit
from skore import CrossValidationReport
from skore_hub_project import Project
from skore_hub_project.artefact.serializer import Serializer
from skore_hub_project.media import (
    EstimatorHtmlRepr,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    RocTest,
    RocTrain,
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


def serialize(object: CrossValidationReport) -> tuple[bytes, str]:
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

    def test_splits_test_samples_density(self, payload):
        assert payload.splits == [
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        ]

    def test_splits_test_samples_density_many_rows(self):
        X, y = make_regression(random_state=42, n_samples=10_000)
        cvr = CrossValidationReport(
            LinearRegression(),
            X,
            y,
            splitter=ShuffleSplit(random_state=42, n_splits=7),
        )
        payload = CrossValidationReportPayload(
            project=Project("<tenant>", "<name>"),
            report=cvr,
            key="<key>",
        )
        splits = payload.splits
        assert len(splits) == 7
        assert all(len(s) == 200 for s in splits)
        for s in splits:
            assert all(bucket >= 0 and bucket <= 1 for bucket in s)

    def test_class_names(self, payload):
        assert payload.class_names == ["1", "0"]

    def test_classes(self, payload):
        X, y = make_classification(
            random_state=42,
            n_samples=10_000,
            n_classes=2,
        )
        cvr = CrossValidationReport(
            LogisticRegression(),
            X,
            y,
            splitter=ShuffleSplit(random_state=42, n_splits=7),
        )
        payload = CrossValidationReportPayload(
            project=Project("<tenant>", "<name>"),
            report=cvr,
            key="<key>",
        )
        classes = payload.classes
        assert len(classes) == 200
        assert np.unique(classes).tolist() == [0, 1]
        assert np.sum(classes) == 93

    def test_classes_many_rows(self, payload):
        assert payload.classes == [0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

    def test_groups(self):
        rng = np.random.default_rng(seed=42)
        X, y = make_regression(random_state=42, n_samples=10_000)

        cvr = CrossValidationReport(
            LinearRegression(),
            X,
            y,
            splitter=GroupKFold(n_splits=7),
            groups=rng.integers(8, size=10_000),
        )
        payload = CrossValidationReportPayload(
            project=Project("<tenant>", "<name>"),
            report=cvr,
            key="<key>",
        )
        groups = payload.groups
        assert len(groups) == 200
        assert np.unique(groups).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert groups[:10] == [6, 3, 6, 3, 5, 3, 2, 0, 7, 6]

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
        assert list(map(type, payload.related_items)) == [
            EstimatorHtmlRepr,
            PrecisionRecallTest,
            PrecisionRecallTrain,
            RocTest,
            RocTrain,
        ]

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
